import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage } from "langchain/schema";

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

const model = new ChatOpenAI({
  temperature: 0.9,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

app.get("/", async (req: Request, res: Response) => {
  const response = await model.call([
    new HumanMessage("Dime como te llamas?")
  ]);
  res.send(response.content);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
