## To launch the chatbot as a web service

1. $ clone ...
2. $ cd into chatbot_for_cloudstrap
3. $ docker-compose up
4. Download the weights archive and unzip in the 'weights' folder

### Testing


$ curl localhost:5000?question=hi%20there
<br>
{
    "response": "good to see you"
}
