import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { LLMChain } from "langchain/chains";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

const model = new ChatOpenAI({
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Indicaciones iniciales al modelo
const systemTemplate =
  "Eres un asistente que traduce del {input_language} al {output_language}.";
const systemMessagePrompt =
  SystemMessagePromptTemplate.fromTemplate(systemTemplate);

// Mensaje del usuario
const humanTemplate = "{text}";
const humanMessagePrompt =
  HumanMessagePromptTemplate.fromTemplate(humanTemplate);

const chatPrompt = ChatPromptTemplate.fromPromptMessages([
  systemMessagePrompt,
  humanMessagePrompt,
]);

app.get("/", async (req: Request, res: Response) => {
  const chain = new LLMChain({
    llm: model,
    prompt: chatPrompt,
  });
  // Las variables indicadas en "{}" se agregan como parámetros en el método call
  const result = await chain.call({
    input_language: "Español",
    output_language: "Italiano",
    text: "Me encanta programar",
  });
  res.send(result.text);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
