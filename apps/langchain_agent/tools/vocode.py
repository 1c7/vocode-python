import logging
import os
from langchain.agents import tool
from dotenv import load_dotenv

from vocode.streaming.models.message import BaseMessage
from call_transcript_utils import delete_transcript, get_transcript

load_dotenv()

from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
import time


@tool("call phone number")
def call_phone_number(input: str) -> str:
    """calls a phone number as a bot and returns a transcript of the conversation.
    the input to this tool is a pipe separated list of a phone number, a prompt, and the first thing the bot should say.
    The prompt should instruct the bot with what to do on the call and be in the 3rd person,
    like 'the assistant is performing this task' instead of 'perform this task'.

    for example, `+15555555555|the assistant is explaining the meaning of life|i'm going to tell you the meaning of life` will call +15555555555, say 'i'm going to tell you the meaning of life', and instruct the assistant to tell the human what the meaning of life is.
    """
    phone_number, prompt, initial_message = input.split("|", 2)
    call = OutboundCall(
        base_url=os.getenv("TELEPHONY_SERVER_BASE_URL"),
        to_phone=phone_number,
        from_phone=os.getenv("OUTBOUND_CALLER_NUMBER"),
        config_manager=RedisConfigManager(),
        agent_config=ChatGPTAgentConfig(
            prompt_preamble=prompt,
            end_conversation_on_goodbye=True,
            initial_message=BaseMessage(text=initial_message),
        ),
        logger=logging.Logger("call_phone_number"),
    )
    call.start()
    while True:
        maybe_transcript = get_transcript(call.conversation_id)
        if maybe_transcript:
            delete_transcript(call.conversation_id)
            return maybe_transcript
        else:
            time.sleep(1)
