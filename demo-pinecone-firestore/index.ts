import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
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
import { initializeApp, cert } from "firebase-admin/app";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

// Initialize Firebase
const serviceAccount = require("./serviceAccountKey.json");
initializeApp({
  credential: cert(serviceAccount),
});

// Construimos una conversacion previa
const previousConversation = () => {
  const chatHistory = new FirestoreChatMessageHistory({
    collectionName: "langchain",
    sessionId: "session user 3",
    userId: "id user 3",
    config: serviceAccount,
  });
  chatHistory.addUserMessage(
    "Hola, mi nombre es Carlos Huamani y soy un programador informático"
  );
  chatHistory.addAIChatMessage(
    "Hola, mucho gusto en conocerte. Yo soy una inteligencia artificial"
  );
  chatHistory.addUserMessage(
    "Yo tengo 31 años y me gusta programar en JavaScript"
  );
  chatHistory.addAIChatMessage("Yo tengo 1 año y me gusta programar en Python");
};

// Logica para conseguir la conversacion previa
const getConversation = async () => {
  const chatHistory = new FirestoreChatMessageHistory({
    collectionName: "langchain",
    sessionId: "session user 3",
    userId: "id user 3",
    config: serviceAccount,
  });

  const memory = new BufferMemory({
    memoryKey: "history_current",
    chatHistory,
  });
  return memory;
};

// Logica para conseguir los documentos
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

// Logica para conseguir el vector
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

// Logica para crear el indice
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
      memories: memoryConversation
        ? [memoryVector, memoryConversation]
        : [memoryVector],
    }),
  });

  // Ejecutar el chain
  const res = await chain.call({
    input: "Cual es mi nomnbre?",
  });
  return res.text;
};

app.get("/", async (req: Request, res: Response) => {
  const response = await getAnswer();
  res.send(response);
});

app.get("/previous-conversation", async (req: Request, res: Response) => {
  await previousConversation();
  res.send("success");
});

app.get("/index-docs", async (req: Request, res: Response) => {
  const response = await createIndex();
  res.send(response);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
