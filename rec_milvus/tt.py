import os  # noqa
import json
# Get an access_token through personal access token oroauth.
coze_api_token = "pat_sQ4WPeAbOiFhJ2kbVrnEV3ORuCyUr38U49XemHGVYGacdUp0tbwjHxLQwDVMB1lP"

from cozepy import (
    COZE_CN_BASE_URL,
    BotPromptInfo,
    ChatEventType,
    Coze,
    DeviceOAuthApp,
    Message,
    MessageContentType,
    TokenAuth,
    setup_logging,
)

# Init the Coze client through the access_token.
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=COZE_CN_BASE_URL)

# Create a bot instance in Coze, copy the last number from the web link as the bot's ID.
bot_id = "7494672904324857910"
# The user id identifies the identity of a user. Developers can use a custom business ID
# or a random string.
user_id = "8429894984"

# Call the coze.chat.stream method to create a chat. The create method is a streaming
# chat and will return a Chat Iterator. Developers should iterate the iterator to get
# chat event and handle them.

chat_poll = coze.chat.create_and_poll(
    bot_id=bot_id,
    user_id=user_id,
    additional_messages=[
        Message.build_user_question_text("https://www.xinjiang.gov.cn/xinjiang/xwtt/202504/908168fc3cee479cae10b86a7e9c4134.shtml"),
        Message.build_assistant_answer("请你按照json格式的数据返回给我，title、time、content.")
    ],
)
