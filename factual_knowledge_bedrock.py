import boto3
import os
import glob
import json

from fmeval.data_loaders.data_config import DataConfig
from bedrock_claud3_model_runner import BedrockClaude3ModelRunner
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig

# Bedrock clients for model inference
bedrock = boto3.client(service_name='bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')

model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
# model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'

config = DataConfig(
    dataset_name="factual_knowledge",
    dataset_uri="data/factual_knowledge.jsonl",
    dataset_mime_type=MIME_TYPE_JSONLINES,
    model_input_location="question",
    target_output_location="answers"
)

content_template = '{"anthropic_version": "bedrock-2023-05-31", "system": "Provide the answer in one sentence and less than 5 words", "prompt": $prompt, "max_tokens": 500}'

bedrock_model_runner = BedrockClaude3ModelRunner(
    model_id=model_id,
    output='text',
    content_template=content_template
)

eval_algo = FactualKnowledge(FactualKnowledgeConfig(target_output_delimiter="<OR>"))
eval_output = eval_algo.evaluate(model=bedrock_model_runner, dataset_config=config, 
                                 prompt_template="$model_input", save=True)
                                 

eval_response = json.dumps(eval_output, default=vars, indent=4)
eval_response_json = json.loads(eval_response)
output_path = eval_response_json[0]["output_path"]

with open(output_path, "r") as file:
    for line in file:
        print(line)