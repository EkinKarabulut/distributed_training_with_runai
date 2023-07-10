# Base image
FROM tensorflow/tensorflow:2.10.1-gpu

# Set the working directory
WORKDIR /app

# Copy the Python script and other required files
COPY launch.sh utils.py distributed.py /app/

# Set the command to run when the container starts
# Run the bash file
RUN chmod u+x launch.sh
CMD ["./launch.sh"]

