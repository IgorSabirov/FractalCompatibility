Как использовать
# How to Build Your Own Analyzer
1. **Collect Data**:
   - Use a smartphone camera (720p, 30 fps) for eye tracking and HRV.
   - Record audio (44.1 kHz) for voice.
2. **Process**:
   - Use MediaPipe for microsaccades.
   - Use Librosa for voice features.
   - Use rPPG for HRV.
3. **Compare**:
   - Run `analyzer.py` to generate and compare personality codes.
4. **Improve**:
   - Train a neural network (CNN-LSTM) on large datasets (EyeLink, AudioSet, PhysioNet).
​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
