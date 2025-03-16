#!/bin/bash

# Install FFmpeg without "sudo" (Streamlit Cloud doesn't allow sudo)
apt-get update -y
apt-get install -y ffmpeg

# Verify installation
ffmpeg -version