from sentence_transformers import SentenceTransformer, util
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, List, Optional, Literal
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from urllib.parse import urljoin
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from diskcache import Cache
import requests
import logging
import hashlib
import json
import os

load_dotenv()

# ----------------------------------
# Logging Configuration
# ----------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:  %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------
# Nvidia Configuration
# ----------------------------------
def get_nvidia_llm():
    return ChatNVIDIA(
        model="nvidia/nemotron-4-340b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.2,
        top_p=0.7,
        stream=False,
    )


# Initialize the Bedrock-based language model
llm = get_nvidia_llm()

# SentenceTransformer for vector-based similarity
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------
# Define State for LangGraph
# ----------------------------------
# Define the state structure for the workflow, representing user input, selected options, and outputs.
class AgentState(TypedDict):
    user_input: str
    selected_level: Optional[str]
    selected_school: Optional[str]
    selected_program_links: Optional[List[str]]
    scraped_data: Optional[str]
    answer: Optional[str]
    reflection: Optional[str]
    revised_answer: Optional[str]
    memory: List[dict]
    feedback: Optional[bool]
    final: Optional[bool]


# ----------------------------------
# Utility Utilities
# ----------------------------------
# Fetch all available program links from the university catalog website.
def getPrograms():

    base_url = "https://catalog.odu.edu/programs/"

    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract program links categorized by undergraduate and graduate levels
    program_links = []
    for li_tag in soup.find_all("li"):
        a_tag = li_tag.find("a", href=True)
        if a_tag:
            href = a_tag["href"]
            if "/undergraduate/" in href or "/graduate/" in href:
                full_url = urljoin(base_url, href)
                program_links.append(full_url)

    # Remove duplicates from the program links
    program_links = list(set(program_links))
    schools = {
        "Arts & Letters": "/arts-letters/",
        "Business": "/business/",
        "Education": "/education/",
        "Sciences": "/sciences/",
        "Engineering & Technology": "/engineering-technology/",
        "Health Sciences": "/health-sciences/",
        "Nursing": "/nursing/",
        "Interdisciplinary": [
            "/cybersecurity/",
            "/data-science/",
            "/supply-chain-logistics-maritime-operations/",
        ],
    }

    # Categorize links by level and school
    categorized_links = {
        "undergraduate": {school: [] for school in schools.keys()},
        "graduate": {school: [] for school in schools.keys()},
    }

    for link in program_links:
        level = "undergraduate" if "/undergraduate/" in link else "graduate"
        for school, pattern in schools.items():
            if isinstance(pattern, list):
                if any(p in link for p in pattern):
                    categorized_links[level][school].append(link)
                    break
            elif pattern in link:
                categorized_links[level][school].append(link)
                break

    return categorized_links


cache = Cache("./cache_dir", eviction_policy="least-recently-used")


def generate_cache_key(links: List[str]) -> str:
    """Generates a hash key for the given links."""
    return hashlib.sha256("".join(links).encode()).hexdigest()


# Vector-based similarity search
def vector_search(query: str, links: List[str]) -> List[str]:
    """
    Perform vector-based similarity search between a query and a list of links.
    """
    # Encode links
    link_vectors = {}
    for link in links:
        try:
            link_vectors[link] = embedding_model.encode(
                link, batch_size=10, show_progress_bar=False
            )
        except Exception as e:
            None

    # Encode query
    try:
        query_vector = embedding_model.encode(
            query, batch_size=10, show_progress_bar=False
        )
    except Exception as e:
        return []

    # Calculate similarity scores
    scores = {
        link: util.cos_sim(query_vector, vector).item()
        for link, vector in link_vectors.items()
    }

    # Log top matches for debugging
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Filter links by a lower threshold
    threshold = 0.5  # Adjust this value as needed
    matched_links = [link for link, score in sorted_scores if score > threshold]

    return matched_links


# Scrape program details from program links.
def scrape_program_links(links: List[str]) -> str:

    data = []
    for link in links:
        response = requests.get(link)
        response.raise_for_status()  # Raise an error if the request fails

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract tab contents and organize them into structured data
        tab_data = {}
        tabs = soup.select('a[role="tab"]')  # Select tab elements
        for tab in tabs:
            tab_name = tab.text.strip()
            tab_id = tab.get("aria-controls")
            if tab_id:
                content = soup.find(id=tab_id)
                if content:
                    tab_data[tab_name] = content.get_text(strip=True)

        documents = [
            {"tab": tab, "content": content} for tab, content in tab_data.items()
        ]

        # Prompt for formatting data using the LLM
        generate_prompt = PromptTemplate(
            template="""You are an assistant for data formatting. You are given a raw text from a web page containing information about a university program. \n
                The data contains tables with courses required for that program and other sorts of information. Your task is to format and organize the raw data given into readable information. \n
                Think carefully.\n
                Page Data: {documents} \n
                Page Data Formatted:""",
            input_variables=["question", "documents"],
        )

        rag_chain = generate_prompt | llm | StrOutputParser()

        generation = rag_chain.invoke({"documents": documents})
        data.append(generation)
        logger.info(f"Pages done: {len(data)}")
    return "\n".join(data)


# ----------------------------------
# Workflow Agents
# ----------------------------------
# Define routing agent for determining level, school, and program.
def routing_agent(state: AgentState) -> AgentState:
    """Uses the LLM to determine level, school, and specific program, then verifies against PROGRAM_LINKS."""
    PROGRAM_LINKS = getPrograms()  # Fetch categorized program links
    user_input = state["user_input"]

    # Use LLM to suggest routing
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an academic advisor AI for Old Dominion University (ODU)."
                "Based on the user's query and using the current conversation history, determine the following:"
                "\n1. Program level: 'undergraduate' or 'graduate'."
                "\n2. School: Select from ['Arts & Letters', 'Business', 'Education', 'Sciences', 'Engineering & Technology', "
                "'Health Sciences', 'Nursing', 'Interdisciplinary']."
                "\n3. Specific program name as closely as possible (e.g., 'Data Science BS', 'Nursing MS')."
                "Return only a JSON object with the keys: 'level', 'school', 'program'. No note, preamble or explanation will be accepted.",
            ),
            (
                "human",
                "Question: {user_input}, Current Conversation History (You are AI, they are User): {memory}",
            ),
        ]
    )
    conversation_history = "\n".join(
        f"User: {msg['content']}" if msg["type"] == "human" else f"AI: {msg['content']}"
        for msg in state["memory"]
    )
    response = llm.invoke(
        prompt.format_messages(user_input=user_input, memory=conversation_history)
    )
    try:
        # Parse the LLM's response and validate links
        logger.info(response.content)
        parsed_response = json.loads(response.content)
        level = (
            parsed_response.get("level", "").lower()
            if parsed_response.get("level")
            else None
        )
        school = (
            parsed_response.get("school", "") if parsed_response.get("school") else None
        )
        program = (
            parsed_response.get("program", "").lower()
            if parsed_response.get("program")
            else None
        )

        # Verify Program Exists in Suggested School
        filtered_links = []
        if level in PROGRAM_LINKS and school in PROGRAM_LINKS[level]:
            school_links = PROGRAM_LINKS[level][school]
            filtered_links = vector_search(program, school_links)

            # Fallback Search if No Links Found in Suggested School
            if not filtered_links:
                for sch, links in PROGRAM_LINKS[level].items():
                    filtered_links += vector_search(program, links)

        # Final Validation
        if filtered_links:
            logger.info(
                f"Routing Agent: Level: {level.capitalize()}, School: {school or 'Using Multiple'}, Program: {program}, Links: {len(filtered_links)} found"
            )
            state.update(
                {
                    "selected_level": level,
                    "selected_school": school,
                    "selected_program_links": filtered_links,
                }
            )
        else:
            state["selected_program_links"] = []

    except Exception as e:
        logger.error(f"Routing Agent Error: {e}")
        state["selected_program_links"] = []

    return state


def data_processing_agent(state: AgentState) -> AgentState:
    """Uses the scraper function to organize data with caching."""
    links = state["selected_program_links"]
    cache_key = generate_cache_key(links)

    # Check if cached data exists
    if cache_key in cache:
        logger.info("Cache hit: Returning cached data.")
        scraped_data = cache[cache_key]
    else:
        logger.info("Cache miss: Scraping data.")
        scraped_data = scrape_program_links(links)
        # Cache the result with a time-to-live (TTL) of 1 week (7 days)
        cache.set(cache_key, scraped_data, expire=7 * 24 * 60 * 60)  # 7 days
        state["scraped_data"] = scraped_data
    return state


# Generate a first answer to the question using scraped or cached data and memory
def question_answering_agent(state: AgentState) -> AgentState:
    """Generates an answer based on the scraped data."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an academic advisor AI specializing in ODU programs answering a User. When talking about classes, always add the class number to the name. Use the conversation history for additional context.",
            ),
            (
                "human",
                "{question}\nRelevant Data:\n{data}\nCurrent Conversation Memory (You are AI, they are User):\n{memory}",
            ),
        ]
    )

    conversation_history = "\n".join(
        f"User: {msg['content']}" if msg["type"] == "human" else f"AI: {msg['content']}"
        for msg in state["memory"]
    )
    formatted_prompt = prompt.format_messages(
        question=state["user_input"],
        data=state["scraped_data"],
        memory=conversation_history,
    )
    response = llm.invoke(formatted_prompt)
    state["answer"] = response.content
    return state


# Reflect on the first answer to better it if needed
def reflection_agent(state: AgentState) -> AgentState:
    """Reflects on the answer based on the scraped data."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an academic advisor AI specializing in ODU programs. You critic answers to advising questions.",
            ),
            (
                "human",
                "Reflect on the answer given to this question: {question}\nAnswer: {answer}",
            ),
        ]
    )

    formatted_prompt = prompt.format_messages(
        question=state["user_input"], answer=state["answer"]
    )
    response = llm.invoke(formatted_prompt)
    state["reflection"] = response.content
    return state


# Revise the first answer using the reflection it made on it with the scraped/cached data
def revision_agent(state: AgentState) -> AgentState:
    """Revises the answer based on feedback."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an academic advisor AI specializing in ODU programs."
                "Revise the given answer so that it provides **only the final answer** without any reasoning, preamble, or additional explanations."
                "Use the feedback provided as well as the relevant data, you also have feedback scores and reasons from previous questions and answers where 1 is the worst answer possible and 5 is a perfectr one"
                "The response should begin immediately with the information requested.",
            ),
            (
                "human",
                "Revise the answer given to this question: {question}\n"
                "Answer: {answer}\n"
                "Feedback for revision: {reflection}\n"
                "Relevant Data:\n{data}\n"
                "Previous Feedback:\n{feedback_history}",
            ),
        ]
    )
    # Ensure the file exists and is properly initialized
    if (
        not os.path.exists("./data/feedback_data.json")
        or os.path.getsize("./data/feedback_data.json") == 0
    ):
        # Create the file and initialize it with an empty list
        with open("./data/feedback_data.json", "w") as f:
            json.dump([], f)

    # Load existing data safely
    try:
        with open("./data/feedback_data.json", "r") as f:
            feedback_list = json.load(f)  # Load existing JSON data
    except json.JSONDecodeError:
        logger.warning(
            f"{"./data/feedback_data.json"} is invalid. Reinitializing as empty."
        )
        feedback_list = []  # Reset to empty if the file is malformed
    feedback_history = "\n".join(
        f"User: {feedback['question']}\nAI: {feedback['answer']}\nFeedback Score: {feedback['feedback']}\nReason for Score: {feedback['reason']}"
        for feedback in feedback_list
    )
    formatted_prompt = prompt.format_messages(
        question=state["user_input"],
        answer=state["answer"],
        reflection=state["reflection"],
        data=state["scraped_data"],
        feedback_history=feedback_history,
    )
    response = llm.invoke(formatted_prompt)
    state["revised_answer"] = response.content
    return state


# Checking if answer correrctly answers the question
def feedback_validation_agent(state: AgentState) -> AgentState:
    """Performs feedback validation."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an academic advisor AI specializing in ODU programs. You answer strictly only using the integers 1, 2, 3, 4 and 5.",
            ),
            (
                "human",
                "You are given a question and an answer, you also have feedback from prior question and answers. Using the feedback from those prior questions and answers as a base, score this question and answer between 1 and 5 with 1 being the worst answer possible and 5 being a perfect one.\nQuestion: {question}\nAnswer: {revised_answer}\nPrior Feedback: {feedback}",
            ),
        ]
    )
    # Load existing data safely
    try:
        with open("./data/feedback_data.json", "r") as f:
            feedback_list = json.load(f)  # Load existing JSON data
    except json.JSONDecodeError:
        logger.warning(
            f"{"./data/feedback_data.json"} is invalid. Reinitializing as empty."
        )
        feedback_list = []  # Reset to empty if the file is malformed

    feedback_history = "\n".join(
        f"User: {feedback['question']}\nAI: {feedback['answer']}\nFeedback Score: {feedback['feedback']}\nReason for Score: {feedback['reason']}"
        for feedback in feedback_list
    )
    formatted_prompt = prompt.format_messages(
        question=state["user_input"],
        revised_answer=state["revised_answer"],
        feedback=feedback_history,
    )
    response = llm.invoke(formatted_prompt)

    validation_result = int(response.content.strip())
    return {"feedback": validation_result}


def feedback_checker(state):
    if state["feedback"] >= 4:
        logger.info(
            f"Feedback Validation Agent: Predicted feedback score is {state["feedback"]}"
        )
        return "feedback_pass"
    else:
        logger.warning(
            f"Feedback Validation Agent: Predicted feedback score is {state["feedback"]}, going back to revision."
        )
        return "feedback_fail"


# Checking if answer correrctly answers the question
def final_validation_agent(state: AgentState) -> AgentState:
    """Performs final validation."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an academic advisor AI specializing in ODU programs. You answer strictly only in True or False.",
            ),
            (
                "human",
                "Check if the answer answers correctly to the question, if it does then answer True if not then False.\nQuestion: {question}\nAnswer: {revised_answer}",
            ),
        ]
    )

    formatted_prompt = prompt.format_messages(
        question=state["user_input"],
        revised_answer=state["revised_answer"],
    )
    response = llm.invoke(formatted_prompt)

    validation_result = response.content.strip().lower()
    return {"final": validation_result}


def final_checker(state):
    if "true" in state["final"]:
        logger.info("Final Validation Agent: Answer is valid. Ending workflow.")
        return "true"
    else:
        logger.warning(
            "Final Validation Agent: Answer is invalid. Returning to question_answerer."
        )
        return "false"


# ----------------------------------
# Build LangGraph Workflow
# ----------------------------------
# Define a multi-step workflow using LangGraph.
class GraphConfig(TypedDict):
    model_name: Literal["anthropic"]


def build_workflow():
    workflow = StateGraph(AgentState, config_schema=GraphConfig)

    # Define the nodes in the workflow
    workflow.add_node("router", routing_agent)
    workflow.add_node("data_processor", data_processing_agent)
    workflow.add_node("question_answerer", question_answering_agent)
    workflow.add_node("reflector", reflection_agent)
    workflow.add_node("reviser", revision_agent)
    workflow.add_node("feedback_checker", feedback_validation_agent)
    workflow.add_node("final_checker", final_validation_agent)

    # Set entry point and define edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "data_processor")
    workflow.add_edge("data_processor", "question_answerer")
    workflow.add_edge("question_answerer", "reflector")
    workflow.add_edge("reflector", "reviser")
    workflow.add_edge("reviser", "feedback_checker")

    # Add conditional edges for reviser
    workflow.add_conditional_edges(
        "feedback_checker",
        feedback_checker,
        {
            "feedback_pass": "final_checker",
            "feedback_fail": "reviser",
        },
    )

    # Add conditional edges for feedback checker
    workflow.add_conditional_edges(
        "final_checker",
        final_checker,
        {
            "true": END,
            "false": "question_answerer",
        },
    )

    return workflow.compile()
