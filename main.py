import whisperx
import gc
import torch
import time
import aiohttp
import openai
import asyncio
import os

hf_token="USE YOUR HG TOKEN"
openai_api_key = "USE YOUR OPENAI KEY"
audio_file = r"Audio file path"
transcription_file_path = r"transcription file path"
output_folder = r"summary output path"
summaryfilename="summary2.txt"

# Asynchronous function to read the transcription
async def read_transcription(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Asynchronous function to summarize transcription with OpenAI
async def summarize_transcription_with_openai(full_transcription, openai_api_key):
    openai.api_key = openai_api_key
    try:
        response = await asyncio.to_thread(openai.chat.completions.create,
            model="gpt-4",  # GPT-4 model
            messages=[{
                "role": "system",
                "content": "You are the best summarizer in the world, capable of identifying key points, speaker traits, and conversation themes with high accuracy."
            }, {
                "role": "user",
                "content": f"""
                Please analyze the transcription content provided below and generate a summary of around 300 words. Each heading containing corresponding relevant points, including:
                1. **Key Points**: Extract the most important points discussed in the conversation.
                2. **Speaker Information**: Identify if any speaker consistently **stressed** particular points or concepts, and explain which parts of the conversation they focused on.
                3. **Tone Analysis**: Describe the **tone** of each speaker (e.g., assertive, empathetic, informative, inquisitive) based on their speech patterns.
                4. **Conversation Theme**: Determine the main theme(s) of the conversation (e.g., learning, business, personal topics, etc.).
                5. **Contextual Insights**: Identify any interesting or noteworthy context in the conversation that adds to its meaning or significance.
                Transcription content:
                {full_transcription}
                """
            }],
            max_tokens=310,  
            temperature=0.5
        )

        summary = response.choices[0].message.content
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_path = os.path.join(output_folder, summaryfilename)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(summary)
        print(f"Summary saved to {file_path}")
        return summary
    except Exception as e:
        print("Error summarizing transcription with OpenAI:", str(e))
        return None


async def process_transcription(audio_file, transcription_file_path, openai_api_key):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)

    gc.collect()
    torch.cuda.empty_cache()

    start_time = time.time()
    # Transcription
    result = model.transcribe(audio, batch_size=4)
    print("BEFORE ALIGNMENT:\n", result)

    # Alignment
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print("AFTER ALIGNMENT:\n", result)

    # Diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="hf_token", device=device)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=3)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(result["segments"])

    # Save conversation output
    with open(transcription_file_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f"Start: {segment['start']}, End: {segment['end']}, Speaker: {segment['speaker']}\n")
            f.write(f"Text: {segment['text']}\n")
            f.write("\n")
    print(f"Conversation saved to {transcription_file_path}")

    # Summarize transcription
    full_transcription = await read_transcription(transcription_file_path)
    summary = await summarize_transcription_with_openai(full_transcription, openai_api_key)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for transcription and diarization: {elapsed_time:.2f} seconds")
    
    if summary:
        print("Summary of the transcription:\n", summary)
    else:
        print("Could not summarize the transcription.")

async def main():
    await process_transcription(audio_file, transcription_file_path, openai_api_key)

if __name__ == "__main__":
    asyncio.run(main())
