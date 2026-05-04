# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

"""
Workforce example using Gemini models from Google.

To run this file, you need to configure the Google Gemini API key.
You can obtain your API key from Google AI Studio: https://aistudio.google.com/
Set it as GOOGLE_API_KEY="your-api-key" in your .env file or add it to your environment variables.

Changelog:
  v1  Migration GEMINI_3_PRO -> GEMINI_2_0_FLASH
  v1  WORKFORCE_MODE flag + construct_society() bridge (F-01)
  v1  max_tokens: 4096 quota protection (F-04)
  v2  OPTIMIZATION -- Error 429 fix (Free Tier limits)
      - Disabled Document Processing Agent & Image Analysis Agent
      - Disabled Reasoning Coding Agent
      - Reduced to 2 model instances (web_model + orchestrator_model)
      - Shared web_model for BrowserToolkit (browsing + planning)
      - max_tokens reduced 4096 -> 2048
      - ROUND_LIMIT = 5 -> 3 (max_steps confinement)
      - GEMINI_2_0_FLASH -> GEMINI_2_0_FLASH_LITE (GEMINI_1_5_FLASH invalid)
      - headless=True forced on BrowserToolkit
"""

import sys
import pathlib
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.toolkits import (
    FunctionTool,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    BrowserToolkit,
    FileToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level
from camel.tasks.task import Task

from camel.societies import Workforce

from owl.utils import DocumentProcessingToolkit

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# MODULE FLAG: signals to webapp.py that this module uses the Workforce
# paradigm (NOT the RolePlaying paradigm).  webapp.py checks this flag to
# decide whether to call run_society() or to invoke step() directly.
# ---------------------------------------------------------------------------
WORKFORCE_MODE = True
ROUND_LIMIT = 3  # max_steps confinement (was 5, was 15 originally)

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_agent_list() -> List[Dict[str, Any]]:
    """Construct a reduced agent list to minimize RPM / token usage.

    Active agents (2):
      - Web Agent            (web_model)

    Disabled agents (commented out to reduce model instances from 8 to 1):
      - Document Processing Agent
      - Image Analysis Agent
      - Reasoning Coding Agent
    """
    # -----------------------------------------------------------------
    # Single shared model instance  (was 8 separate instances before v2)
    # -----------------------------------------------------------------
    web_model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_2_0_FLASH_LITE,
        model_config_dict={"temperature": 0, "max_tokens": 2048},
    )

    # -----------------------------------------------------------------
    # Toolkits  (only what the Web Agent needs)
    # -----------------------------------------------------------------
    search_toolkit = SearchToolkit()

    # Reuse web_model for document processing (avoids extra model instance)
    document_processing_toolkit = DocumentProcessingToolkit(model=web_model)

    # BrowserToolkit: headless=True to save resources;
    # web_model shared for both browsing_model and planning_agent_model
    browser_toolkit = BrowserToolkit(
        headless=True,
        web_agent_model=web_model,
        planning_agent_model=web_model,
    )

    # -----------------------------------------------------------------
    # DISABLED toolkits (commented out -- not needed by Web Agent alone)
    # -----------------------------------------------------------------
    # image_analysis_toolkit = ImageAnalysisToolkit(model=...)  # REMOVED
    # code_runner_toolkit  = CodeExecutionToolkit(...)           # REMOVED
    # file_toolkit         = FileToolkit()                       # REMOVED
    # excel_toolkit        = ExcelToolkit()                      # REMOVED

    # -----------------------------------------------------------------
    # Web Agent  (sole active worker)
    # -----------------------------------------------------------------
    web_agent = ChatAgent(
        """You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.
Keep in mind that:
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.
- If one way fails to provide an answer, try other ways or methods. The answer does exist.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.
- You can also simulate browser actions to get more information or verify the information you have found.
- Browser simulation is also helpful for finding target URLs. Browser simulation operations do not necessarily need to find specific answers, but can also help find web page URLs that contain answers (usually difficult to find through simple web searches). You can find the answer to the question by performing subsequent operations on the URL, such as extracting the content of the webpage.
- Do not solely rely on document tools or browser simulation to find the answer, you should combine document tools and browser simulation to comprehensively process web page information. Some content may need to do browser simulation to get, or some content is rendered by javascript.
- In your response, you should mention the urls you have visited and processed.

Here are some tips that help you perform web search:
- Never add too many keywords in your search query! Some detailed results need to perform browser interaction to get, not using search toolkit.
- If the question is complex, search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding official sources rather than direct answers.
  For example, as for the question "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website?", your first search term must be coarse-grained like "National Geographic YouTube" to find the youtube website first, and then try other fine-grained search terms step-by-step to find more urls.
- The results you return do not have to directly answer the original question, you only need to collect relevant information.
""",
        model=web_model,
        tools=[
            FunctionTool(search_toolkit.search_duckduckgo),
            FunctionTool(search_toolkit.search_wiki),
            FunctionTool(document_processing_toolkit.extract_document_content),
            *browser_toolkit.get_tools(),
        ],
    )

    # -----------------------------------------------------------------
    # DISABLED agents  (kept commented for future reactivation)
    # -----------------------------------------------------------------
    # document_processing_agent = ChatAgent(...)   # DISABLED -- saves 1 model
    # reasoning_coding_agent      = ChatAgent(...)   # DISABLED -- saves 1 model

    agent_list = [
        {
            "name": "Web Agent",
            "description": "A helpful assistant that can search the web, extract webpage content, simulate browser actions, and retrieve relevant information.",
            "agent": web_agent,
        },
        # DISABLED: Document Processing Agent  (uncomment to re-enable)
        # DISABLED: Reasoning Coding Agent      (uncomment to re-enable)
    ]
    return agent_list


def construct_workforce() -> Workforce:
    """Construct a workforce with coordinator and task agents.

    Optimization v2: shared orchestrator_model for both coordinator and task
    agent (was 2 separate instances).  Total model instances in the entire
    module is now 2  (web_model + orchestrator_model).
    """
    # Single shared model for orchestrator layer
    orchestrator_model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_2_0_FLASH_LITE,
        model_config_dict={"temperature": 0, "max_tokens": 2048},
    )

    task_agent = ChatAgent(
        "You are a helpful assistant that can decompose tasks and assign tasks to workers.",
        model=orchestrator_model,
    )

    coordinator_agent = ChatAgent(
        "You are a helpful assistant that can assign tasks to workers.",
        model=orchestrator_model,
    )

    workforce = Workforce(
        "Workforce",
        task_agent=task_agent,
        coordinator_agent=coordinator_agent,
    )

    agent_list = construct_agent_list()

    for agent_dict in agent_list:
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
        )

    return workforce


def construct_society(question: str) -> Any:
    """Construct a Workforce bridge compatible with the Gradio webapp.

    This function is called by webapp.py when the user selects the
    ``run_gemini`` module.  It returns a ``WorkforceWebBridge`` object
    that exposes the same ``init_chat()`` / ``step()`` duck-typing as
    the RolePlaying society objects, so that webapp.py can dispatch
    uniformly regardless of the underlying paradigm.

    The webapp.py detects ``WORKFORCE_MODE = True`` at the module level
    and calls ``society.step()`` directly (instead of the RolePlaying
    ``run_society()`` wrapper which expects a 2-tuple return from
    ``step()``).

    Args:
        question: The user's natural-language question or task prompt.

    Returns:
        A WorkforceWebBridge instance ready for ``init_chat()`` /
        ``step()`` invocation.
    """
    workforce = construct_workforce()

    class WorkforceWebBridge:
        """Adapter that makes a CAMEL Workforce look like an OwlRolePlaying
        society from the perspective of the Gradio webapp."""

        def __init__(self, wf: Workforce, q: str):
            self.wf = wf
            self.q = q
            self.terminated = False

        def init_chat(self, *args, **kwargs):
            """Accept the init_prompt from webapp.py (no-op for Workforce)."""
            return []

        def step(self, *args, **kwargs):
            """Execute the workforce task and return the final answer.

            Returns:
                WorkforceBridgeResponse with ``.msg.content`` and
                ``.terminated = True`` so the webapp can extract the
                answer string.
            """
            if self.terminated:
                return None

            task = Task(content=self.q)
            result = self.wf.process_task(task)
            self.terminated = True

            # Build a response object compatible with webapp.py expectations
            return WorkforceBridgeResponse(result.result if result else "")

    return WorkforceWebBridge(workforce, question)


class WorkforceBridgeResponse:
    """Minimal response object that the webapp can read answer from."""

    def __init__(self, text: str):
        # Provide .msg.content (used by run_society-style readers)
        # and .content as a flat fallback.
        self.msg = type("Msg", (), {"content": text})
        self.content = text
        self.terminated = True


def main():
    r"""Main function to run the OWL system with an example question."""
    # Default research question
    default_task_prompt = "Use Browser Toolkit to summarize the github stars, fork counts, etc. of camel-ai's owl framework, and write the numbers into a python file using the plot package, save it locally, and run the generated python file. Note: You have been provided with the necessary tools to complete this task."

    # Override default task if command line argument is provided
    task_prompt = sys.argv[1] if len(sys.argv) > 1 else default_task_prompt

    task = Task(
        content=task_prompt,
    )

    workforce = construct_workforce()

    processed_task = workforce.process_task(task)

    # Output the result
    print(f"\033[94mAnswer: {processed_task.result}\033[0m")


if __name__ == "__main__":
    main()
