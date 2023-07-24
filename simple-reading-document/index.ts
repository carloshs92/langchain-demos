import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

// Initialize the LLM to use for answering the question.
const model = new OpenAI({
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const responseFromDocument = async () => {
  // Leer el documento
  const loader = new TextLoader("./document.txt");
  const docs = await loader.load();

  // Crear un vector a partir del documento
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  // Crear un retriever a partir del vector
  const retriever = vectorStore.asRetriever();

  // Crear una cadena que use OpenAI LLM y el retriever
  const chain = RetrievalQAChain.fromLLM(model, retriever);

  // Realizar una pregunta a la cadena
  const res = await chain.call({
    query: "¿Quien es Axel?",
  });
  return res.text;
};

app.get("/", async (req: Request, res: Response) => {
  const response = await responseFromDocument();
  res.send(response);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
