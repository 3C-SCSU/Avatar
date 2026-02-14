import os
import time
from PySide6.QtCore import QObject, Slot, Property, Signal
from Developers import devCharts  # Ensure this import stays!

class DevelopersAPI(QObject):
    # Signal to tell QML that the images have been updated
    pathsChanged = Signal()

    def __init__(self):
        super().__init__()
        self._gold_path = ""
        self._silver_path = ""
        self._bronze_path = ""
        self._medal_path = ""
        # Initialize paths on startup
        self.devImagePath()

    @Slot(result=str)
    def getDevList(self):
        return devCharts.devList()

    @Slot(result=str)
    def getTicketsByDev(self) -> str:
        return devCharts.ticketsByDev_text()

    @Slot()
    def devChart(self):
        print("Generating charts...")
        try:
            # 1. Run the generation logic from devCharts.py
            devCharts.main() 
            
            # 2. Update paths with new timestamps and notify QML
            self.devImagePath()
            self.pathsChanged.emit() 
            print("Charts generated and QML notified.")
        except Exception as e:
            print(f"Error generating charts: {e}")

    def devImagePath(self):
        # Anchor to the directory where this API file sits
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Point to the Developers/plotDevelopers subfolder
        plots_dir = os.path.join(base_dir, "Developers", "plotDevelopers")
        
        # Unique timestamp forces QML to bypass its image cache
        timestamp = int(time.time())
        
        def format_path(filename):
            full_p = os.path.abspath(os.path.join(plots_dir, filename)).replace("\\", "/")
            # Use file:/// for QML local file access
            return f"file:///{full_p}?t={timestamp}"

        self._gold_path = format_path("gold_contributors.png")
        self._silver_path = format_path("silver_contributors.png")
        self._bronze_path = format_path("bronze_contributors.png")
        self._medal_path = format_path("Medal.png")

    # Properties with 'notify' decorators so QML updates automatically
    @Property(str, notify=pathsChanged)
    def goldPath(self):
        return self._gold_path

    @Property(str, notify=pathsChanged)
    def silverPath(self):
        return self._silver_path

    @Property(str, notify=pathsChanged)
    def bronzePath(self):
        return self._bronze_path

    @Property(str, notify=pathsChanged)
    def medalPath(self):
        return self._medal_path
