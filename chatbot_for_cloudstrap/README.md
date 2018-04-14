## To launch the chatbot as a web service

1. $ clone ...
2. $ cd into chatbot_for_cloudstrap
3. $ docker-compose up
4. $ mkdir weights
4. Download the weights archive and unzip in the `weights` folder created in the previous step.

The folder should look like this:

<br>chatbot_for_cloudstrap/<br>
 &nbsp;-&nbsp;app_chatbot.py<br>
 &nbsp;-&nbsp;... other files<br>
 &nbsp;-&nbsp;weights/checkpoint<br>
 &nbsp;-&nbsp;weights/seq2seq_model.ckpt-43000.data-00000-of-00001<br>
 &nbsp;-&nbsp;weights/seq2seq_model.index<br>
 &nbsp;-&nbsp;weights/seq2seq_model.meta<br>
  

### Testing


$ curl localhost:5000?question=hi%20there
<br>
{
    "response": "good to see you"
}

## To launch as errbot plugin

The plugin files are in the `errbot-root` subfolder of this repo.
Everything is the same except for some path variable differences.
Don't forget to include the weights in the `/plugins/hipchap/weights` folder