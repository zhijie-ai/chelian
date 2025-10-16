# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/19 10:26
# @User   : RANCHODZU
# @File   : audio.py
# @e-mail : zushoujie@ghgame.cn
from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="The quick brown fox jumped over the lazy dog."
)
response.stream_to_file(speech_file_path)

