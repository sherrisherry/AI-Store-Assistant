FROM sherrizhao/meetfresh-chatbot:v0.0.0
EXPOSE 8002
WORKDIR /app
COPY . .
CMD ["python", "bot.py"]