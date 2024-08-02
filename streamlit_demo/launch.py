from shared.views import App
from shared.utils.log import custom_logger
from shared.utils.pages import set_page_config

set_page_config()
custom_logger()

app = App()
app.view(key="app")
