import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { Client } from "@notionhq/client";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { FirestoreChatMessageHistory } from "langchain/stores/message/firestore";
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

const getConversation = async () => {
  const memory = new BufferMemory({
    memoryKey: "history_current",
    chatHistory: new FirestoreChatMessageHistory({
      collectionName: "langchain",
      sessionId: "lc-example",
      userId: "a@example.com",
      config: { projectId: "your-project-id" },
    }),
  });
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

const queryVector = async () => {
  // Buscar el vector
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY || "",
    environment: process.env.PINECONE_ENVIRONMENT_ONBOARDING || "",
  });
  const pineconeIndex = client.Index(
    process.env.PINECONE_INDEX_ONBOARDING || ""
  );

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
  const docs = await getDocuments(
    process.env.NOTION_ONBOARDING_DOCUMENTS_ID || ""
  );
  const text = await textSplitter.splitDocuments(docs);

  // Crear un vector a partir del documento
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY || "",
    environment: process.env.PINECONE_ENVIRONMENT_ONBOARDING || "",
  });
  const pineconeIndex = client.Index(
    process.env.PINECONE_INDEX_ONBOARDING || ""
  );
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

  // Obtener vector
  const retriever = await queryVector();

  // Buscar conversaciones
  const memoryVector = new VectorStoreRetrieverMemory({
    vectorStoreRetriever: retriever,
  });

  const memoryConversation = await getConversation();

  // Crear estructura del prompt
  const prompt = PromptTemplate.fromTemplate(`
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
      memories: [memoryVector, memoryConversation],
    }),
  });

  // Ejecutar el chain
  const res = await chain.call({
    input: "Quien es Carlos Huamani?",
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
