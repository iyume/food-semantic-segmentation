import sys

from PyQt5.QtWidgets import QApplication

from UI.layout import View

app = QApplication(sys.argv)
# font_id = QFontDatabase.addApplicationFont("NotoEmoji-Regular.ttf")
# font_sid = QFontDatabase.applicationFontFamilies(font_id)[0]
# app.setFont(QFont("JetBrainsMono NFM"))
# app.setFont(QFont("Noto Emoji"))
view = View()
view.show()
sys.exit(app.exec())
