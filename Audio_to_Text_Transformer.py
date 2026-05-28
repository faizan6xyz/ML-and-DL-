"""
Audio to Text Conversion using Transformers (Speech-to-Text)
Supports multiple transformer models for NLP-based audio transcription
"""

import os
import librosa
import numpy as np
import torch
import warnings
from typing import Dict, List, Union
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# APPROACH 1: Using OpenAI Whisper (Recommended)
# ============================================================================

class WhisperAudioTransformer:
    """
    Whisper-based Speech-to-Text Transformer
    Supports multiple languages and model sizes
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper transformer
        
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
        """
        try:
            import whisper
            self.whisper = whisper
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(model_size, device=self.device)
            print(f"✓ Whisper model '{model_size}' loaded on {self.device}")
        except ImportError:
            print("Install whisper: pip install openai-whisper")
            raise
    
    def transcribe(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio/video file
            language: ISO-639-1 code (e.g., 'en', 'es', 'fr') or None for auto-detect
            
        Returns:
            Dictionary with transcription and metadata
        """
        try:
            result = self.model.transcribe(audio_path, language=language)
            return {
                'text': result['text'],
                'language': result.get('language'),
                'segments': result.get('segments', []),
                'model': 'whisper'
            }
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return None
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Get transcription with word-level timestamps
        
        Args:
            audio_path: Path to audio/video file
            
        Returns:
            List of segments with timestamps
        """
        result = self.model.transcribe(audio_path)
        timestamps = []
        
        for segment in result['segments']:
            timestamps.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
        
        return timestamps


# ============================================================================
# APPROACH 2: Using Wav2Vec2 (Facebook/Meta)
# ============================================================================

class Wav2Vec2AudioTransformer:
    """
    Wav2Vec2-based Speech-to-Text Transformer
    Fast, lightweight, and accurate for English
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Initialize Wav2Vec2 transformer
        
        Args:
            model_name: Hugging Face model identifier
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print(f"✓ Wav2Vec2 model loaded on {self.device}")
        except ImportError:
            print("Install transformers: pip install transformers")
            raise
    
    def transcribe(self, audio_path: str, sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio using Wav2Vec2
        
        Args:
            audio_path: Path to audio file
            sample_rate: Audio sample rate (default 16kHz)
            
        Returns:
            Dictionary with transcription
        """
        try:
            # Load audio
            speech_array, _ = librosa.load(audio_path, sr=sample_rate)
            
            # Process audio
            with torch.no_grad():
                inputs = self.processor(
                    speech_array, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                )
                logits = self.model(
                    inputs.input_values.to(self.device)
                ).logits
            
            # Decode prediction
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return {
                'text': transcription,
                'model': 'wav2vec2',
                'language': 'en'
            }
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return None


# ============================================================================
# APPROACH 3: Using Hugging Face Speech-to-Text Models
# ============================================================================

class HuggingFaceSpeechTransformer:
    """
    Generic Hugging Face Speech-to-Text Pipeline
    Supports multiple pre-trained models
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-large-xlsr-53-english"):
        """
        Initialize speech recognition pipeline
        
        Args:
            model_name: Hugging Face model identifier
        """
        try:
            from transformers import pipeline
            self.device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=self.device
            )
            print(f"✓ Speech-to-Text pipeline initialized with {model_name}")
        except ImportError:
            print("Install transformers: pip install transformers")
            raise
    
    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio using pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription
        """
        try:
            result = self.pipe(audio_path)
            return {
                'text': result['text'],
                'model': 'huggingface-asr',
                'confidence': result.get('confidence', 'N/A')
            }
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class AudioProcessor:
    """Utility class for audio processing and preparation"""
    
    @staticmethod
    def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Audio array
        """
        speech_array, _ = librosa.load(audio_path, sr=sr)
        return speech_array
    
    @staticmethod
    def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_audio_path: Output audio file path (optional)
            
        Returns:
            Path to extracted audio
        """
        try:
            from moviepy.editor import VideoFileClip
            
            if output_audio_path is None:
                output_audio_path = video_path.replace('.mp4', '.wav').replace('.avi', '.wav')
            
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
            print(f"✓ Audio extracted: {output_audio_path}")
            return output_audio_path
        except ImportError:
            print("Install moviepy: pip install moviepy")
            raise
    
    @staticmethod
    def resample_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
        """
        Resample audio to specific sample rate
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        y, _ = librosa.load(audio_path, sr=sr)
        return y
    
    @staticmethod
    def chunk_audio(audio_array: np.ndarray, sr: int = 16000, chunk_duration: int = 30) -> List[np.ndarray]:
        """
        Split audio into chunks
        
        Args:
            audio_array: Audio array
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of audio chunks
        """
        chunk_samples = sr * chunk_duration
        chunks = [audio_array[i:i + chunk_samples] for i in range(0, len(audio_array), chunk_samples)]
        return chunks


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

def example_whisper_transcription():
    """Example: Transcribe audio using Whisper"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Whisper Speech-to-Text")
    print("="*60)
    
    try:
        # Initialize Whisper
        whisper_model = WhisperAudioTransformer(model_size="base")
        
        # Example: Transcribe (replace with your audio file)
        audio_file = "your_audio.mp3"  # or .wav, .m4a, .ogg, etc.
        
        if os.path.exists(audio_file):
            result = whisper_model.transcribe(audio_file)
            print(f"\nTranscription: {result['text']}")
            print(f"Language: {result['language']}")
        else:
            print(f"Note: Audio file '{audio_file}' not found. Provide your own audio file.")
    
    except Exception as e:
        print(f"Error: {e}")


def example_wav2vec2_transcription():
    """Example: Transcribe audio using Wav2Vec2"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Wav2Vec2 Speech-to-Text")
    print("="*60)
    
    try:
        # Initialize Wav2Vec2
        wav2vec_model = Wav2Vec2AudioTransformer()
        
        # Example: Transcribe
        audio_file = "your_audio.wav"
        
        if os.path.exists(audio_file):
            result = wav2vec_model.transcribe(audio_file)
            print(f"\nTranscription: {result['text']}")
        else:
            print(f"Note: Audio file '{audio_file}' not found. Provide your own audio file.")
    
    except Exception as e:
        print(f"Error: {e}")


def example_video_to_text():
    """Example: Convert video to text"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Video to Text Conversion")
    print("="*60)
    
    try:
        video_file = "your_video.mp4"
        
        if os.path.exists(video_file):
            # Extract audio from video
            audio_file = AudioProcessor.extract_audio_from_video(video_file)
            
            # Transcribe audio
            whisper_model = WhisperAudioTransformer(model_size="base")
            result = whisper_model.transcribe(audio_file)
            print(f"\nVideo Transcription: {result['text']}")
        else:
            print(f"Note: Video file '{video_file}' not found. Provide your own video file.")
    
    except Exception as e:
        print(f"Error: {e}")


def example_batch_transcription():
    """Example: Batch transcribe multiple audio files"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Transcription")
    print("="*60)
    
    try:
        whisper_model = WhisperAudioTransformer(model_size="base")
        
        # Process all audio files in a directory
        audio_dir = "./audio_files"
        
        if os.path.exists(audio_dir):
            for audio_file in os.listdir(audio_dir):
                if audio_file.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    audio_path = os.path.join(audio_dir, audio_file)
                    result = whisper_model.transcribe(audio_path)
                    print(f"\n{audio_file}: {result['text']}")
        else:
            print(f"Note: Create '{audio_dir}' directory with audio files.")
    
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUDIO TO TEXT TRANSFORMER - NLP Speech-to-Text")
    print("="*60)
    
    # Uncomment the example you want to run:
    
    # example_whisper_transcription()
    # example_wav2vec2_transcription()
    # example_video_to_text()
    # example_batch_transcription()
    
    # OR use directly in your code:
    # from Audio_to_Text_Transformer import WhisperAudioTransformer
    # model = WhisperAudioTransformer()
    # result = model.transcribe("audio.mp3")
    # print(result['text'])
