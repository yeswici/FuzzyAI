from fuzzy.consts import WIKI_LINK


def generate_model_error(model_type: str, model_name: str, default_model_name: str) -> str:
    return (f"\033[91m{model_type} model: {model_name} was not added to the fuzzer,"
            f" please make sure you add it with -x <provider/model> and set"
            f" -e {model_type.replace(' ', '_')}_model=<provider/model> accordingly"
            f" (you can omit -e if using the default {model_type} model - {default_model_name})."
            f" Attack wiki: {WIKI_LINK}\033[0m")

