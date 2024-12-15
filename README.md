# FuzzyAI Fuzzer
<div style="text-align: center;">
<img src="resources/logo.png" alt="Project Logo" width="200" style="vertical-align:middle; margin-right:10px;" />
</div>
<br><br>

The FuzzyAI Fuzzer is a powerful tool for automated LLM fuzzing. It is designed to help developers and security researchers identify and mitigate potential security vulnerabilities in their LLM APIs. 

![FZAI](resources/fuzz.gif)

## Features

- **Fuzzing Techniques**: The FuzzyAI Fuzzer supports various fuzzing techniques, including mutation-based fuzzing, generation-based fuzzing, and intelligent fuzzing.
- **Input Generation**: It provides built-in input generation capabilities to generate valid and invalid inputs for testing.
- **Integration**: The FuzzyAI Fuzzer can be easily integrated into existing development and testing workflows.
- **Extensibility**: It provides an extensible architecture, allowing users to customize and extend the fuzzer's functionality.

## Attacks we already implemented

| Attack Type                                  | Title                                                                                                                                                                       | Reference                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| ArtPrompt                                    | ASCII Art-based jailbreak attacks against aligned LLMs                                                                                                                      | [arXiv:2402.11753](https://arxiv.org/pdf/2402.11753)                            |
| Taxonomy-based paraphrasing                  | Uses persuasive language techniques like emotional appeal and social proof to jailbreak LLMs                                                                                | [arXiv:2401.06373](https://arxiv.org/pdf/2401.06373)                            |
| PAIR (Prompt Automatic Iterative Refinement) | Automates the generation of adversarial prompts by pairing two LLMs (“attacker” and “target”) to iteratively refine prompts until achieving jailbreak                       | [arXiv:2310.08419](https://arxiv.org/pdf/2310.08419)                            |
| Many-shot jailbreaking                       | Exploits large context windows in language models by embedding multiple fake dialogue examples, gradually weakening the model's safety responses                            | [Anthropic Research](https://www.anthropic.com/research/many-shot-jailbreaking) |
| Genetic                                      | Genetic algorithm iteratively modifies prompts to generate an adversarial suffix that coerces large language models into producing restricted content.                      | [arXiv:2309.01446](https://arxiv.org/pdf/2309.01446)                            |
| Hallucinations                               | Using Hallucinations to Bypass RLHF Filters                                                                                                                                 | [arXiv:2403.04769](https://arxiv.org/pdf/2403.04769.pdf)                        |
| DAN (Do Anything Now)                        | Promotes the LLM to adopt an unrestricted persona that ignores standard content filters, allowing it to "Do Anything Now".                                                  | [GitHub Repo](https://github.com/0xk1h0/ChatGPT_DAN)                            |
| WordGame                                     | Disguises harmful prompts as word puzzles                                                                                                                                   | [arXiv:2405.14023](https://arxiv.org/pdf/2405.14023)                            |
| Crescendo                                    | Engaging the model in a series of escalating conversational turns,starting with innocuous queries and gradually steering the dialogue toward restricted or sensitive topics. | [arXiv:2404.01833](https://arxiv.org/pdf/2404.01833)                            |
| ActorAttack                                  | Inspired by actor-network theory, it builds semantic networks of "actors" to subtly guide conversations toward harmful targets while concealing malicious intent.           | [arxiv 2410.10700](https://arxiv.org/pdf/2410.10700)                                                                            |                                                                                                                                     |
| Back To The Past                             | Modifies the prompt by adding a profession-based prefix and a past-related suffix                                                                                           |                                                                                 |
| Please                                       | Modifies the prompt by adding please as a prefix and suffix                                                                                                                   |                                                                                 |
| Thought Experiment                           | Modifies the prompt by adding a thought experiment-related prefix. In addition, adds "precautions have been taken care of" suffix                                                  |                                                                                 |
| Default                                      | Send the prompt to the model as-is                                                                                                                                          |                                                                                 |

## Supported models
We've tested the attacks above using the models listed below. FuzzyAI is programmed to support a definitive set of model types,
but you can always add support for new models [by adding the model's name to the relevant LLM provider](wiki/diy/new_models.html).

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Anthropic</strong></td>
      <td>
        <ul>
          <li>claude-3-5-sonnet-latest</li>
          <li>claude-3-opus-latest</li>
          <li>claude-3-haiku-20240307</li>
          <li>claude-2.1</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><strong>OpenAI</strong></td>
      <td>
        <ul>
          <li>o1-preview</li>
          <li>o1-mini</li>
          <li>gpt-4o</li>
          <li>gpt-4-turbo</li>
          <li>gpt-4</li>
          <li>gpt-3.5-turbo</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><strong>Gemini</strong></td>
      <td>
        <ul>
          <li>gemini-1.5-pro</li>
          <li>gemini-pro</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><strong>Azure</strong></td>
      <td>
        <ul>
          <li>gpt-4o</li>
          <li>gpt-4</li>
          <li>gpt-35-turbo</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><strong>Bedrock</strong></td>
      <td>
        <ul>
          <li>anthropic.claude-3-5-sonnet-20241022-v2:0</li>
          <li>anthropic.claude-3-sonnet-20240229-v1:0</li>
          <li>anthropic.claude-3-opus-20240229-v1:0</li>
          <li>anthropic.claude-3-haiku-20240307-v1:0</li>
          <li>anthropic.claude-v2:1</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><strong>AI21</strong></td>
      <td>
        <ul>
          <li>jamba-1.5-mini</li>
          <li>jamba-1.5-large</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><strong>Ollama</strong></td>
      <td>
        <ul>
          <li>llama3.2</li>
          <li>llama3.1</li>
          <li>dolphin-llama3</li>
          <li>llama3</li>
          <li>llama2:70b</li>
          <li>llama2-uncensored</li>
          <li>llama2</li>
          <li>vicuna</li>
          <li>gemma2</li>
          <li>gemma</li>
          <li>phi3</li>
          <li>phi</li>
          <li>mistral</li>
          <li>mixtral</li>
          <li>qwen</li>
          <li>zephyr</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Adding support for newer models
Please note that we have specified the models tested for cloud API. If you attempt to use a model not listed, you will receive an error indicating that the provider does not support that model. However, <a href="diy/new_models.html">you can add the model to the implementation's list of supported models</a>, and it will then function as expected.

## Supported Cloud APIs
- **OpenAI**
- **Anthropic**
- **Gemini**
- **Azure Cloud**
- **AWS Bedrock**
- **AI21**
- **Huggingface ([Downloading models](https://huggingface.co/docs/hub/en/models-downloading))**
- **Ollama**
- **Custom REST API**

## Datasets
We've included a few datasets you can use, they're to be found under the resources/ folder
<br>
<span style="color:#cc6666;">Note:</span> Some of the prompts may be grammatically incorrect; this is intentional, as it appears to be more effective against the models.

| File name                                     | Description                                                                                              |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------|
| pandoras_prompts.txt                          | Harmful prompts                                                                         |
| adv_prompts.txt                               | Harmful prompts                                                                                          |
| benign_prompts.txt                            | Regular prompts                                                                                          |
| history_prompts.txt                           | Harmful prompts phrased as in "Back To The Past" attack                                           |
| harmful_behaviors.csv                         | Harmful prompts                                                                                          |
| adv_suffixes.txt                              | Random prompt suffixes                                                                                   |
| alpaca_data_instructions.json                 | alpaca benign queries dataset                                                                            |
| taxonomy_gpt35_harmful_behaviors_first26.json | persuasive prompts                                                                                       |
| finetuned_summarizer_train_dataset.jsonl      | Dataset used to train a GPT fine-tuned summarizer (See [Paper](https://arxiv.org/pdf/2401.06373) page 20) 

## Documentation

For more detailed instructions, please refer to the [documentation](wiki/index.html).

## Getting Started

To get started with the FuzzyAI Fuzzer, follow these steps:

1. Clone the repository: `git clone git@github.com:cyberark/FuzzyAI.git`
2. Install the required dependencies using [poetry](https://python-poetry.org/) : `poetry update`<br/>
this will create a [venv](https://docs.python.org/3/library/venv.html), if you're not using an IDE, make sure you activate it by invoking `poetry shell`
3. Run the fuzzer: `python run.py -h`

## Usage

Just run the following and follow the flags
```bash
python run.py -h
```

Note: To run models using OLLAMA, make sure it is installed or download it [here](https://ollama.com/). 
Follow the instructions to pull the desired model.


## Examples

### A default attack (just evaluate the prompt 'Harmful_Prompt') using mistral over Ollama
```bash
python run.py -m ollama/mistral -a def -t "Harmful_Prompt"
```

### Attack the prompt utilizing ManyShot and Taxonomy attacks, using gpt3.5 over OpenAI using one worker
```bash
OPENAI_API_KEY=sk-3fa1... python run.py -m openai/gpt-3.5-turbo -a man -a tax -t "Harmful_Prompt"
```

### Attack prompts in prompts.txt utilizing the ArtPrompt attack, using Claude 3.5 over Anthropic
```bash
ANTHROPIC_API_KEY=deadbeef... python run.py -m anthropic/claude-3-5-sonnet-20240620 -a art -T resources/adv_prompts.txt -e blacklisted_words="bomb,garbage,acid,ugly,self-harm,kill your"
```

## Persisting Your Settings
To save your configuration, you can create a JSON-formatted config file where the keys correspond to the long-form command-line flags. For example, see [config_example.json](config_example.json):
```json
{
  "model": [
    "ollama/mistral"
  ],
  "attack_modes": [
    "def",
    "art"
  ],
  "classifier": [
    "har"
  ],
  "extra": [
    "blacklisted_words=acid"
  ]
}
```
Once you've customized the configuration to your needs, you can apply these settings by running the following command:prev
```bash
python run.py -C config_example.json -t "Harmful_Prompt"
```

## Caveats
* Some classifiers do more than just evaluate a single output. For example, the cosine-similarity classifier compares two outputs by measuring the angle between them, while a 'harmfulness' classifier checks whether a given output is harmful. As a result, not all classifiers are compatible with the attack methods we've implemented, as those methods are designed for single-output classifiers.
* When using the -m option with OLLAMA models, <b>ensure that all OLLAMA models are added first before adding any other models.</b> Use the -e port=... option to specify the port number for OLLAMA (default is 11434).

## Contributing

Contributions are welcome! If you would like to contribute to the FuzzyAI Fuzzer, please follow the guidelines outlined in the [CONTRIBUTING.md](https://github.com/cyberark/FuzzyAI/blob/main/CONTRIBUTING.md) file.

## License

The FuzzyAI Fuzzer is released under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](https://github.com/cyberark/FuzzyAI/blob/main/LICENSE) file for more details.

## Contact

If you have any questions or suggestions regarding the FuzzyAI Fuzzer, please feel free to contact us at [fzai@cyberark.com](mailto:fzai@cyberark.com).

