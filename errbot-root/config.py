import logging

# This is a minimal configuration to get you started with the Text mode.
# If you want to connect Errbot to chat services, checkout
# the options in the more complete config-template.py from here:
# https://raw.githubusercontent.com/errbotio/errbot/master/errbot/config-template.py

BACKEND = 'Hipchat'  # Errbot will start in text mode (console only mode) and will answer commands from there.

BOT_ALT_PREFIXES=('@hipchap', '@hipguy', 'Err',)

BOT_DATA_DIR = r'/Users/aram/errbot-root/data'
BOT_EXTRA_PLUGIN_DIR = r'/Users/aram/errbot-root/plugins'

BOT_LOG_FILE = r'/Users/aram/errbot-root/errbot.log'
BOT_LOG_LEVEL = logging.DEBUG

BOT_ADMINS = ('@aram', '121531_....@chat.hipchat.com', )  # !! Don't leave that to "@CHANGE_ME" if you connect your errbot to a chat system !!

BOT_IDENTITY = {
    'username' : '121531_xxxxxx8@chat.hipchat.com',
    # Group admins can create/view tokens on the settings page after logging
    # in on HipChat's website
    'password' : 'G......H.......!',
    'token' : 'XNoWshYbpfcaVfEh1wS5Vy2muPWEyL2JIfbSC5Qw'
}

CHATROOM_FN = "Hip Chap"
CHATROOM_PRESENCE = ('121531_bot_tests',)

#XMPP_CA_CERT_FILE = "/etc/ssl/certs/ca-certificates.crt"
XMPP_CA_CERT_FILE = None
