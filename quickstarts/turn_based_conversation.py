import logging
from dotenv import load_dotenv
from vocode import getenv
from vocode.helpers import create_microphone_input_and_speaker_output
from vocode.turn_based.agent.chat_gpt_agent import ChatGPTAgent
from vocode.turn_based.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.turn_based.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.turn_based.transcriber.whisper_transcriber import WhisperTranscriber
from vocode.turn_based.turn_based_conversation import TurnBasedConversation

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

# See https://api.elevenlabs.io/v1/voices
ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

if __name__ == "__main__":
    microphone_input, speaker_output = create_microphone_input_and_speaker_output(
        streaming=False, use_default_devices=False
    )

    conversation = TurnBasedConversation(
        input_device=microphone_input,
        output_device=speaker_output,
        transcriber=WhisperTranscriber(api_key=getenv("OPENAI_API_KEY")),
        agent=ChatGPTAgent(
            system_prompt="你是一个投资机构的智能 AI，你的任务是作为面试官去考核创业者，第一个问题：请你简单介绍你的项目，后续的问题针对项目继续追问",
            initial_message="你好, 我们同事有没有给你介绍过面试流程？就是10分钟的快问快答。",
            api_key=getenv("OPENAI_API_KEY"),
        ),
        synthesizer=AzureSynthesizer(
            api_key=getenv("AZURE_SPEECH_KEY"),
            region=getenv("AZURE_SPEECH_REGION"),
            # voice_name="en-US-SteffanNeural",
            voice_name="zh-CN-YunyangNeural",
        ),
        logger=logger,
    )
    print("开始对话. 按 Ctrl+C 退出.")
    while True:
        try:
            input("按回车键开始录音...")
            conversation.start_speech()
            input("按回车键结束录音...")
            conversation.end_speech_and_respond()
        except KeyboardInterrupt:
            break
