import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain } from "langchain/chains";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

const createIndex = async () => {
  // Leer el documento
  const loader = new TextLoader("./document.txt");
  const docs = await loader.load();

  // Crear un vector a partir del documento
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY || "",
    environment: process.env.PINECONE_ENVIRONMENT || "",
  });
  const pineconeIndex = client.Index(process.env.PINECONE_INDEX || "");
  await PineconeStore.fromDocuments(docs, new OpenAIEmbeddings(), {
    pineconeIndex,
  });
  return "indexed.!";
};

const getAnswer = async () => {
  // Initialize the LLM to use for answering the question.
  const model = new OpenAI({
    temperature: 0.9,
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

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

  // Crear la cadena a partir del modelo y del vector
  const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
    k: 1,
    returnSourceDocuments: true,
  });

  // Realizar una pregunta a la cadena
  const response = await chain.call({ query: "¿Quien es Carlos?" });

  return response.text;
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
