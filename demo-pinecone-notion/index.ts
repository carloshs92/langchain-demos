import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { Client } from "@notionhq/client";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { NotionAPILoader } from "langchain/document_loaders/web/notionapi";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { LLMChain } from "langchain/chains";
import {
  BufferMemory,
  CombinedMemory,
  VectorStoreRetrieverMemory,
} from "langchain/memory";
import { PromptTemplate } from "langchain/prompts";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

const createBufferMemory = async () => {
  const memory = new BufferMemory({
    memoryKey: "history_current",
  });
  await memory.chatHistory.addUserMessage("Me llamo Carlos");
  await memory.chatHistory.addAIChatMessage("Mucho gusto");
  return memory;
};

const getDocuments = async (id: string) => {
  const pageLoader = new NotionAPILoader({
    clientOptions: {
      auth: process.env.NOTION_API_KEY || "",
    },
    id,
    type: "page",
  });

  const pageDocs = await pageLoader.loadAndSplit();

  return pageDocs;
};

const getInstructions = async () => {
  const notion = new Client({
    auth: process.env.NOTION_API_KEY,
  });

  const notionPages: any = await notion.blocks.children.list({
    block_id: process.env.NOTION_INSTRUCTIONS_ID || "",
  });

  const text = notionPages.results
    ?.map((block: any) => {
      let paragraph = "";
      if (block.type === "paragraph") {
        paragraph = block?.paragraph?.rich_text
          .map((rt: any) => rt.plain_text)
          .join("\n");
      }
      return paragraph;
    })
    .join("\n\n");

  return text;
};

const queryVector = async () => {
  // Buscar el vector
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY || "",
    environment: process.env.PINECONE_ENVIRONMENT || "",
  });
  const pineconeIndex = client.Index(process.env.PINECONE_INDEX || "");

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );
  return vectorStore.asRetriever(10);
};

const createIndex = async () => {
  // Crate text splitter
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
  });

  // Leer el documento
  const docs = await getDocuments(process.env.NOTION_DOCUMENTS_ID || "");
  const text = await textSplitter.splitDocuments(docs);

  // Crear un vector a partir del documento
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY || "",
    environment: process.env.PINECONE_ENVIRONMENT || "",
  });
  const pineconeIndex = client.Index(process.env.PINECONE_INDEX || "");
  await PineconeStore.fromDocuments(text, new OpenAIEmbeddings(), {
    pineconeIndex,
  });

  return "success";
};

const getAnswer = async () => {
  // Create LLM
  const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  // Load documents
  const instruction = await getInstructions();

  // Obtener vector
  const retriever = await queryVector();

  // Buscar conversaciones
  const memory = new VectorStoreRetrieverMemory({
    vectorStoreRetriever: retriever,
    memoryKey: "history_initial",
  });

  await memory.saveContext({ input: instruction }, { output: "Esta bien" });
  const conversationFromDB = await createBufferMemory();

  // Crear estructura del prompt
  const prompt = PromptTemplate.fromTemplate(`
    Conversación inicial:
    {history_initial}

    Conversación actual:
    {history_current}

    Siguiente pregunta
    Usuario: {input}
    Respuesta IA:`);

  // Crear el chain
  const chain = new LLMChain({
    llm: model,
    prompt,
    memory: new CombinedMemory({
      memories: [conversationFromDB, memory],
    }),
  });

  // Ejecutar el chain
  const res = await chain.call({
    input: "Quien es carlos Huamani?",
  });
  return res.text;
};

app.get("/", async (req: Request, res: Response) => {
  const response = await getAnswer();
  res.send(response);
});

app.get("/index-docs", async (req: Request, res: Response) => {
  const response = await createIndex();
  res.send(response);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
