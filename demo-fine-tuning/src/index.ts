import dotenv from "dotenv";
import { ChatOpenAI } from "langchain/chat_models/openai";
import OpenAI from "openai";
import { HumanMessage, SystemMessage } from "langchain/schema";
import fs from "fs";

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const useFineTunedModel = async (modelName: string) => {
  const model = new ChatOpenAI({
    temperature: 0.5,
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName,
  });
  return model;
};

const createFineTuningJob = async (fileId: string) => {
  // Crear un job de fine-tuning
  const fineTune = await openai.fineTuning.jobs.create({
    training_file: fileId,
    model: "gpt-3.5-turbo-0613",
  });
  return fineTune;
};

const uploadJSONL = async () => {
  const file = await openai.files.create({
    file: fs.createReadStream("mydata.jsonl"),
    purpose: "fine-tune",
  });
  return file;
};

const retrieveFineTuningJob = async (fineTuningID: string) => {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const fineTune = await openai.fineTuning.jobs.retrieve(fineTuningID);
  return fineTune.finished_at ? fineTune : "Aún falta";
};

const init = async () => {
  const action = process.env.action;
  if (action === "upload") {
    const response = await uploadJSONL();
    console.log(response);
  }
  if (action === "retrieve") {
    const id = process.env.npm_config_id as string;
    const response = await retrieveFineTuningJob(id);
    console.log(response);
  }
  if (action === "create") {
    const id = process.env.npm_config_id as string;
    const response = await createFineTuningJob(id);
    console.log(response);
  }
  if (action === "ask") {
    const model = process.env.npm_config_model as string;
    const tunedModel = await useFineTunedModel(model);
    const response = await tunedModel.call([
      new SystemMessage("Asume el rol del asistente Rob"),
      new HumanMessage("Hola en que me puedes ayudar?"),
    ]);
    console.log(response);
  }
};

init();
