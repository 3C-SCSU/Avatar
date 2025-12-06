import os
from PySide6.QtCore import QObject, Slot, Property
import pandas as pd

from Developers import devCharts

class DevelopersAPI(QObject):

    def __init__(self):
        super().__init__()
        self._gold_path = ""
        self._silver_path = ""
        self._bronze_path = ""
        self._medal_path = ""
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
            # Get the data
            data = devCharts.run_shortlog_all()
            if not data:
                print("No contributors found")
                return

            exclude = ["3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>"]
            data = [(n, c) for (n, c) in data if n not in exclude]

            # Assign tiers
            tiered = devCharts.assign_fixed_tiers(data)

            # Generate charts for each tier
            base_dir = os.path.dirname(os.path.abspath(__file__))
            plots_dir = os.path.join(base_dir, "plotDevelopers")
            os.makedirs(plots_dir, exist_ok=True)

            for tier in ["Gold", "Silver", "Bronze"]:
                chart_path = os.path.join(plots_dir, f"{tier.lower()}_contributors.png")
                devCharts.plot_single_tier(tiered, tier, chart_path)
                print(f"Generated {tier} chart")

            # Update paths after generating
            self.devImagePath()
            print("Charts generated successfully")

        except Exception as e:
            print(f"Error generating charts: {e}")


    def devImagePath(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"The base directory path is {base_dir}")
        plots_dir = os.path.join(base_dir, "plotDevelopers")
        print(f"The plots directory path is {plots_dir}")
        gold_path = os.path.abspath(os.path.join(plots_dir, "gold_contributors.png"))
        silver_path = os.path.abspath(os.path.join(plots_dir, "silver_contributors.png"))
        bronze_path = os.path.abspath(os.path.join(plots_dir, "bronze_contributors.png"))
        medal_path =  os.path.abspath(os.path.join(plots_dir, "Medal.png"))

        self._gold_path = "file:///" + gold_path.replace("\\", "/")
        self._silver_path = "file:///" + silver_path.replace("\\", "/")
        self._bronze_path = "file:///" + bronze_path.replace("\\", "/")
        self._medal_path = "file:///" + medal_path.replace("\\", "/")


        print(f"Gold chart path: {self._gold_path}")
        print(f"Silver chart path: {self._silver_path}")
        print(f"Bronze chart path: {self._bronze_path}")
        print(f"Medal path: {self._medal_path}")

        return gold_path, silver_path, bronze_path

    @Property(str, constant = True)
    def goldPath(self):
        return self._gold_path

    @Property(str, constant = True)
    def silverPath(self):
        return self._silver_path

    @Property(str, constant = True)
    def bronzePath(self):
        return self._bronze_path

    @Property(str, constant = True)
    def medalPath(self):
        return self._medal_path
