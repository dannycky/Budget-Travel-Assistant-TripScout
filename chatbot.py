"""
Travel Information Chatbot - RAG Backend
=========================================
Loads markdown travel documents from the `md/` folder, builds a retrieval
index, and exposes a LangGraph-powered RAG chatbot for answering travel
questions with grounded context.
"""

import os
from datetime import date, datetime
from glob import glob
from typing import Annotated, TypedDict, List, Optional

import requests as requests_lib
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# =====================================================================
# Configuration
# =====================================================================
MD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_K = 6

SYSTEM_PROMPT = """You are a helpful travel information assistant for budget travellers,
with access to three tools: live flight search, live hotel search, and a
local travel guide knowledge base.

Always be concise, friendly, and practical.

--- TOOL ROUTING RULES (follow strictly) ---
Choose the tool based on what the user is asking:

1. FLIGHTS: If the user asks about flights, airfares, or flying between
   cities -> use search_google_flights (SerpApi live data).

2. HOTELS: If the user asks about hotels, accommodation, hostels, or
   places to stay -> use search_hotels (SerpApi live data).

3. ACTIVITIES / KNOWLEDGE: If the user asks about sightseeing, attractions,
   activities, restaurants, food, commute/transport tips, neighborhoods,
   or any destination knowledge -> use search_travel_guides (local guides).

4. DEFAULT: If you are unsure which tool fits the question, call
   search_travel_guides first. It is safe, free, and fast.

5. COMBINATIONS: For trip planning ("plan my Tokyo trip"), chain multiple
   tools: flights + hotels + travel guides as needed.

6. DATE-WINDOW COMPARISON (when the user gives a flexible period, e.g.
   "5 days of leave in mid-November", "sometime in October", "which
   week is cheapest"):
   - MAXIMUM 6 WINDOW COMPARISONS per request. Never exceed this cap.
   - Pick up to 6 distinct, sensible windows spread across the user's
     stated period (e.g. early / mid / late in the month, or consecutive
     weeks). Prefer midweek start dates (Tue-Thu) when the user has no
     day preference, as they are usually cheaper.
   - Run the comparison using whichever search type matches the request:
     (a) FLIGHT-ONLY comparison: call search_google_flights once per
         window. For multi-day trips use type="1" with return_date set
         to (window start + trip length - 1 day) so the price is the
         round-trip total.
     (b) HOTEL-ONLY comparison: call search_hotels once per window with
         check_in_date = window start and check_out_date = window start
         + (number of nights). Use sort_by="3" and compare the total
         stay prices.
     (c) FLIGHT + HOTEL comparison: for each window, call both tools
         (one flight search + one hotel search), then combine:
         total cost = round-trip flight price + (hotel per-night price
         x number of nights). Rank the windows by total cost and
         recommend the cheapest.
   - Present the comparison as a clear table or list showing each
     window's flight cost, hotel cost, and combined total, then state
     the cheapest window explicitly. Include the booking links for the
     recommended window.
--- End of Tool Routing Rules ---

--- FLIGHT SEARCH TOOL RULES ---
When the user asks about flights, fares, or prices between cities/airports:
1. ALWAYS convert city names into their 3-letter IATA airport codes before
   calling the search_google_flights tool (e.g. "Tokyo" -> "HND" or "NRT",
   "Hong Kong" -> "HKG", "Osaka" -> "KIX", "London" -> "LHR").
2. Today's date is {today}. Resolve ALL dates against today's date:
   - If the user omits the year (e.g. "23/9" or "September 23"), assume the
     NEXT upcoming occurrence of that date, which is never in the past.
   - If the user says a relative date like "next Friday", work out the exact
     calendar date from today's date.
   - NEVER use a date earlier than today. Flight searches for past dates fail.
3. Dates must be in YYYY-MM-DD format.
4. Use type="2" for one-way flights and type="1" for round-trip flights.
   For round-trips, always provide return_date.
5. After receiving flight results, summarize the best options clearly
   with airline, price, departure time, and duration. ALWAYS include the
   "View/book these flights on Google Flights" link from the tool result
   so the user can book.
--- End of Flight Search Rules ---

--- HOTEL SEARCH TOOL RULES ---
When the user asks about hotels, accommodation, or places to stay:
1. Pass the destination as plain text in the q parameter (e.g. "Beijing",
   "Hotels near Tokyo Station"). No IATA codes needed.
2. Today's date is {today}. Resolve ALL dates against today's date:
   - If the user omits the year, assume the NEXT upcoming occurrence.
   - NEVER use a check-in date earlier than today.
   - check_out_date must be AFTER check_in_date.
3. Dates must be in YYYY-MM-DD format.
4. Map the user's needs to the tool parameters:
   - Number of travellers -> adults (and children + children_ages if any)
   - "cheapest" / "budget" -> sort_by="3"
   - "best" / "highest rated" -> sort_by="8"
   - Maximum budget per night -> max_price (e.g. 150)
   - Star preference (e.g. "4-star hotel") -> hotel_class="4"
   - "highly rated" -> rating="8" (4.0+)
   - "free cancellation" -> free_cancellation=true
   - Preferred currency -> currency (e.g. "HKD", "USD")
5. After receiving hotel results, present each hotel with its name,
   price per night, total price, rating, location, and its per-hotel
   "Details/booking" link. ALWAYS include the "View/book these hotels
   on Google Hotels" link from the tool result so the user can browse
   more options and book.
--- End of Hotel Search Rules ---

--- TRAVEL GUIDES TOOL RULES ---
When answering from search_travel_guides results:
1. Ground your answer in the retrieved guide content and mention which
   guide/destination the info came from.
2. If the guides don't cover the question, say so, then offer general
   travel advice from your own knowledge.
3. The guides are curated for budget travellers — prioritise free and
   cheap options when recommending.
--- End of Travel Guides Rules ---"""


# =====================================================================
# Tools
# =====================================================================
@tool
def search_google_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    type: str = "2",
    return_date: Optional[str] = None,
) -> str:
    """Search live flight prices via SerpApi's Google Flights API.

    Args:
        departure_id: 3-letter IATA code of the departure airport (e.g. "HKG").
        arrival_id: 3-letter IATA code of the arrival airport (e.g. "HND").
        outbound_date: Departure date in YYYY-MM-DD format.
        type: "2" for one-way, "1" for round-trip.
        return_date: Return date in YYYY-MM-DD format (required if type is "1").

    Returns:
        A formatted summary of the top 3 best flights (airline, price,
        departure time, duration), or an error message.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY is not set in the environment."

    # --- Date validation (BEFORE calling SerpApi, to save API quota) ---
    today = date.today()
    try:
        outbound = datetime.strptime(outbound_date, "%Y-%m-%d").date()
    except ValueError:
        return f"Error: outbound_date '{outbound_date}' is invalid. Use YYYY-MM-DD format."
    if outbound < today:
        return (f"Error: outbound_date {outbound_date} is in the past (today is "
                f"{today.isoformat()}). Please re-resolve the user's requested dates "
                "against today's date and retry with a future date.")
    if return_date:
        try:
            ret = datetime.strptime(return_date, "%Y-%m-%d").date()
        except ValueError:
            return f"Error: return_date '{return_date}' is invalid. Use YYYY-MM-DD format."
        if ret < outbound:
            return (f"Error: return_date {return_date} is before the outbound_date "
                    f"{outbound_date}. The return must be on or after the departure.")

    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": type,
        "api_key": api_key,
    }
    if return_date:
        params["return_date"] = return_date

    try:
        resp = requests_lib.get("https://serpapi.com/search", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests_lib.exceptions.HTTPError:
        # Don't include the URL (it contains the api_key) in the error message
        return (f"Error: Flight search failed with HTTP status {resp.status_code}. "
                "Check that the airports, dates, and trip type are valid.")
    except Exception as e:
        return f"Error: Flight search request failed: {type(e).__name__}"

    best_flights = data.get("best_flights") or []
    if not best_flights:
        return "No best flights found for this route and date. Try different dates or airports."

    # Parse only the top 3 itineraries to keep the LLM context small
    lines = [f"Top {min(3, len(best_flights))} flights from {departure_id} to {arrival_id} on {outbound_date}:"]
    for i, flight in enumerate(best_flights[:3], start=1):
        airline = flight.get("airline", "Unknown airline")
        price = flight.get("price")
        dep_time = ""
        duration = flight.get("total_duration")
        # Extract departure time & airline from the first leg if available
        legs = flight.get("flights") or []
        if legs:
            first_leg = legs[0]
            airline = first_leg.get("airline", airline)
            dep_time = first_leg.get("departure_airport", {}).get("time", "")
        price_str = f"${price}" if price is not None else "price unavailable"
        dur_str = f"{duration} min" if duration is not None else "duration unknown"
        lines.append(
            f"{i}. {airline} | Price: {price_str} | Departs: {dep_time or 'N/A'} | Duration: {dur_str}"
        )

    # Attach the direct Google Flights URL for this exact search so the user
    # can view/book these flights (free field in the response, no extra API call)
    flights_url = (data.get("search_metadata") or {}).get("google_flights_url")
    if flights_url:
        lines.append(f"View/book these flights on Google Flights: {flights_url}")

    return "\n".join(lines)


@tool
def search_hotels(
    q: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 2,
    children: int = 0,
    children_ages: Optional[str] = None,
    sort_by: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    hotel_class: Optional[str] = None,
    rating: Optional[str] = None,
    free_cancellation: Optional[bool] = None,
    currency: str = "USD",
) -> str:
    """Search live hotel prices via SerpApi's Google Hotels API.

    Args:
        q: Destination to search (e.g. "Beijing", "Hotels near Tokyo Station").
        check_in_date: Check-in date in YYYY-MM-DD format.
        check_out_date: Check-out date in YYYY-MM-DD format (must be after check-in).
        adults: Number of adult guests (default 2).
        children: Number of children (default 0).
        children_ages: Comma-separated ages of children, e.g. "5,8". Must match
            the children count. Required when children > 0.
        sort_by: "3" for lowest price, "8" for highest rating, "13" for most reviewed.
        min_price: Lower bound of price per night (in the requested currency).
        max_price: Upper bound of price per night (user's maximum budget).
        hotel_class: Comma-separated star filter, e.g. "3" (3-star), "4,5" (4-5 star).
        rating: Filter by guest rating: "7" = 3.5+, "8" = 4.0+, "9" = 4.5+.
        free_cancellation: Set true to only show hotels with free cancellation.
        currency: Currency code for returned prices (default "USD").

    Returns:
        A formatted summary of the top 3 hotels (name, price per night, total
        price, rating, location, and a booking link), or an error message.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY is not set in the environment."

    # --- Date validation (BEFORE calling SerpApi, to save API quota) ---
    today = date.today()
    try:
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d").date()
    except ValueError:
        return f"Error: check_in_date '{check_in_date}' is invalid. Use YYYY-MM-DD format."
    try:
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d").date()
    except ValueError:
        return f"Error: check_out_date '{check_out_date}' is invalid. Use YYYY-MM-DD format."
    if check_in < today:
        return (f"Error: check_in_date {check_in_date} is in the past (today is "
                f"{today.isoformat()}). Please re-resolve the user's requested dates "
                "against today's date and retry with a future date.")
    if check_out <= check_in:
        return (f"Error: check_out_date {check_out_date} must be AFTER the "
                f"check_in_date {check_in_date}.")

    params = {
        "engine": "google_hotels",
        "q": q,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "children": children,
        "currency": currency,
        "api_key": api_key,
    }
    # Optional filters - only include when the user's needs require them
    if children > 0:
        if not children_ages:
            return "Error: children_ages is required when children > 0 (e.g. \"5,8\")."
        params["children_ages"] = children_ages
    if sort_by:
        params["sort_by"] = sort_by
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price
    if hotel_class:
        params["hotel_class"] = hotel_class
    if rating:
        params["rating"] = rating
    if free_cancellation:
        params["free_cancellation"] = True

    try:
        resp = requests_lib.get("https://serpapi.com/search", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests_lib.exceptions.HTTPError:
        # Don't include the URL (it contains the api_key) in the error message
        return (f"Error: Hotel search failed with HTTP status {resp.status_code}. "
                "Check that the destination, dates, and filters are valid.")
    except Exception as e:
        return f"Error: Hotel search request failed: {type(e).__name__}"

    properties = data.get("properties") or []
    if not properties:
        return (f"No hotels found for '{q}' on those dates. "
                "Try a different destination, dates, or relax the filters.")

    # Parse only the top 3 hotels to keep the LLM context small
    lines = [f"Top {min(3, len(properties))} hotels in {q} "
             f"({check_in_date} to {check_out_date}, {adults} adult(s)):"]

    # Sort by extracted price when the user wants the cheapest options
    def price_key(p):
        rate = p.get("rate_per_night") or {}
        return rate.get("extracted_lowest", float("inf"))

    ranked = sorted(properties, key=price_key) if sort_by == "3" else properties

    for i, prop in enumerate(ranked[:3], start=1):
        name = prop.get("name", "Unknown hotel")
        stars = prop.get("hotel_class")
        overall_rating = prop.get("overall_rating")
        reviews = prop.get("reviews")

        rate = prop.get("rate_per_night") or {}
        per_night = rate.get("lowest", "price unavailable")
        total = (prop.get("total_rate") or {}).get("lowest", "")

        # Location: use GPS coordinates (Google Hotels doesn't return an address
        # field in the list results; coordinates let the LLM describe the area)
        gps = prop.get("gps_coordinates") or {}
        lat = gps.get("latitude")
        lng = gps.get("longitude")
        location = f"({lat:.4f}, {lng:.4f})" if lat is not None and lng is not None else "N/A"

        # Booking link: Google Hotels listing for this property
        link = prop.get("serpapi_property_details_link") or "no link available"

        extras = []
        if stars:
            extras.append(f"{stars}-star hotel")
        if overall_rating is not None:
            rating_str = f"{overall_rating:.1f}/5"
            if reviews:
                rating_str += f" ({reviews} reviews)"
            extras.append(rating_str)
        if prop.get("free_cancellation"):
            extras.append("free cancellation")

        lines.append(
            f"{i}. {name}"
            + (f" [{' + '.join(extras)}]" if extras else "")
            + f" | Per night: {per_night}"
            + (f" | Total stay: {total}" if total else "")
            + f" | Location: {location}"
            + f" | Details/booking: {link}"
        )

    # Attach the direct Google Hotels URL for this exact search so the user
    # can browse/book these hotels (free field in the response, no extra API call)
    hotels_url = (data.get("search_metadata") or {}).get("google_hotels_url")
    if hotels_url:
        lines.append(f"View/book these hotels on Google Hotels: {hotels_url}")

    return "\n".join(lines)


# =====================================================================
# Document Loading & Indexing
# =====================================================================
def load_markdown_documents(folder: str = MD_FOLDER) -> List[Document]:
    """Load all .md files from the md/ folder into LangChain Documents."""
    docs = []
    for path in sorted(glob(os.path.join(folder, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
    return docs


def build_vectorstore(docs: List[Document]):
    """Split documents into chunks and index them with a local BM25 retriever.

    BM25 is keyword-based, so no embedding API is required (OpenRouter
    does not support the embeddings endpoint).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return BM25Retriever.from_documents(chunks, k=RETRIEVAL_K)


# Module-level holder so the @tool function below can reach the retriever
# that is built once at TravelChatbot startup.
_retriever_holder = {"retriever": None}


@tool
def search_travel_guides(query: str) -> str:
    """Search the local travel guide knowledge base for destination information.

    Covers 60+ cities (Tokyo, Osaka, Seoul, Taipei, Bangkok, Paris, London,
    etc.) with budget-focused content: sightseeing activities (history, art,
    culture, sports, nature), budget restaurants, and commute/transport tips.

    Use this tool for questions about attractions, activities, food,
    getting around, or any destination knowledge. Do NOT use it for live
    flight or hotel prices (use the flight/hotel search tools instead).

    Args:
        query: What to look for, e.g. "free attractions in Seoul",
            "budget eats in Osaka", "how to get around Tokyo".

    Returns:
        Relevant guide excerpts with their source files, or a message
        saying the guides don't cover the query.
    """
    retriever = _retriever_holder["retriever"]
    if retriever is None:
        return "Error: travel guide index is not available."

    hits = retriever.invoke(query)
    if not hits:
        return (f"The travel guides don't cover '{query}'. "
                "Offer general advice from your own knowledge instead.")

    parts = []
    for h in hits:
        source = h.metadata.get("source", "guide")
        parts.append(f"[{source}]\n{h.page_content}")
    return "\n\n".join(parts)


# =====================================================================
# RAG Chatbot (LangGraph)
# =====================================================================
class TravelChatbot:
    """RAG travel chatbot: retrieve relevant guide chunks, then answer."""

    def __init__(self, thread_id: str = "travel-session-1"):
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "Missing API Key! Set OPENROUTER_API_KEY in your environment or .env file."
            )

        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.3,
        )

        # Bind the flight, hotel & travel guide tools to the LLM
        self.tools = [search_google_flights, search_hotels, search_travel_guides]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build retrieval index from md/ folder and expose it to the
        # search_travel_guides tool via the module-level holder
        docs = load_markdown_documents()
        if not docs:
            print(f"Warning: no .md files found in {MD_FOLDER}. "
                  "Chatbot will answer without guide context.")
            self.vectorstore = None
            _retriever_holder["retriever"] = None
        else:
            self.vectorstore = build_vectorstore(docs)
            _retriever_holder["retriever"] = self.vectorstore

        # LangGraph state
        class State(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]

        def retrieve_and_answer(state: State):
            # No always-on retrieval: the LLM now decides when to call
            # search_travel_guides via tool routing rules in the system prompt.
            messages = [SystemMessage(content=SYSTEM_PROMPT.format(
                today=date.today().isoformat(),
            ))]
            messages += state["messages"]

            # Agent loop: let the LLM call tools until it produces a final answer
            response = None
            for _ in range(12):  # cap tool-call rounds (6-window comparisons need up to 12)
                response = self.llm_with_tools.invoke(messages)

                if not response.tool_calls:
                    # No tool requested -> this is the final answer
                    return {"messages": [response]}

                # Execute each requested tool and feed results back
                messages.append(response)
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    try:
                        tool_fn = {t.name: t for t in self.tools}[tool_name]
                        result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        result = f"Error executing tool {tool_name}: {e}"
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            if response is None:
                return {"messages": [AIMessage(content="Sorry, I couldn't generate an answer.")]}
            return {"messages": [response]}

        graph = StateGraph(State)
        graph.add_node("agent", retrieve_and_answer)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)

        self.compiled_graph = graph.compile(checkpointer=MemorySaver())
        self.config = {"configurable": {"thread_id": thread_id}}

    def ask(self, question: str) -> str:
        """Send a question through the graph and return the answer text."""
        events = self.compiled_graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config=self.config,
            stream_mode="values",
        )
        answer = ""
        for event in events:
            if event.get("messages"):
                last = event["messages"][-1]
                if last.type == "ai":
                    answer = last.content
        return answer or "Sorry, I couldn't generate an answer."

    def get_history(self) -> List[BaseMessage]:
        """Return the conversation history for this thread."""
        state = self.compiled_graph.get_state(self.config)
        return state.values.get("messages", []) if state.values else []
