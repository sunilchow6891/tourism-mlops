
# Use a minimal base image with Python 3.9
FROM python:3.9-slim

# Setting the working directory
WORKDIR /app

# Copying the requirements file and installing dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the application script and other necessary files
COPY app.py .

# Creating a non-root user for security (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user     PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Exposing Streamlit's default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
