import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { EPubLoader } from "langchain/document_loaders/fs/epub";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { AIMessage, HumanMessage, SystemMessage } from "langchain/schema";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

// Initialize the LLM to use for answering the question.
const model = new ChatOpenAI({
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const responseFromDocument = async () => {
  // Leer el documento
  const loader = new EPubLoader("./libro.epub");
  const docs = await loader.load();

  // Crear un vector a partir del documento
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docs.filter((d) => !!d.pageContent),
    embeddings
  );

  // Crear un retriever a partir del vector
  const retriever = vectorStore.asRetriever();

  // Crear una cadena que use OpenAI LLM y el retriever

  const QA_PROMPT = `
    Contexto:
    {context}

    Historial:
    {chat_history}
    
    Pregunta: {question}
    Respuesta:`;

  const chain = ConversationalRetrievalQAChain.fromLLM(model, retriever, {
    qaTemplate: QA_PROMPT,
  });

  // Realizar una pregunta a la cadena
  const res = await chain.call({
    question: "¿Quien es Jaskier?",
    context: "Responde todo en verso en español",
    chat_history: [new HumanMessage("Hola, mi nombre es Carlos")],
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
