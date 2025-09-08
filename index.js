import { ChatGroq } from "@langchain/groq";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import readline from "node:readline/promises";
import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const tool = new TavilySearch({
  maxResults: 3,
  topic: "general",
  // includeAnswer: false,
  // includeRawContent: false,
  // includeImages: false,
  // includeImageDescriptions: false,
  // searchDepth: "basic",
  // timeRange: "day",
  // includeDomains: [],
  // excludeDomains: [],
});

const tools = [tool];
const toolNode = new ToolNode(tools);

//initialize the llm
const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxRetries: 2,
  // other params...
}).bindTools(tools);

async function callModel(state) {
  const response = await llm.invoke(state.messages);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

function shouldContinue(state) {
  const lastMessage = state.messages[state.messages.length - 1];

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length > 0) {
    return "tools";
  }
  return "__end__";
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

const app = workflow.compile({ checkpointer });

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = await rl.question("you: ");
    if (userInput === "/bye") {
      break;
    }
    const finalState = await app.invoke(
      {
        messages: [{ role: "user", content: userInput }],
      },
      { configurable: { thread_id: "1" } }
    );
    const lastMessage = finalState.messages[finalState.messages.length - 1];
    console.log("Agent: ", lastMessage.content);
  }

  rl.close();
}

main();
