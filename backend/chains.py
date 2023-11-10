from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from prompt import STAGE_ANALYZER_PROMPT, UTTERANCE_CHAIN_PROMPT, VALIDATION_CHAIN_PROMPT


class TherapyStageAnalyzerChain(LLMChain):
    """Chain to analyze which stage of a therapeutic conversation the conversation should move into."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        therapy_stage_analyzer_prompt_template = STAGE_ANALYZER_PROMPT
        prompt = PromptTemplate(
            template=therapy_stage_analyzer_prompt_template,
            input_variables=["conversation_history"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TherapyConversationChain(LLMChain):
    """Chain to generate the next utterance for the therapeutic conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        therapy_conversation_prompt = UTTERANCE_CHAIN_PROMPT
        prompt = PromptTemplate(
            template=therapy_conversation_prompt,
            input_variables=[
                "name",
                "conversation_history",
                "context"
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TherapyValidationChain(LLMChain):
    """Chain to classify user input into relevant categories for the therapeutic conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        therapy_validation_prompt_template = VALIDATION_CHAIN_PROMPT

        prompt = PromptTemplate(
            template=therapy_validation_prompt_template,
            input_variables=["chat_history"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
