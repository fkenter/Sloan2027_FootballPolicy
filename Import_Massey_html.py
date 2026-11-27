from html.parser import HTMLParser
import re
import pandas as pd


class _MasseyTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.in_detail = False
        self.main_parts: list[str] = []
        self.detail_parts: list[str] = []
        self.current_row: list[tuple[str, str]] = []
        self.rows: list[list[tuple[str, str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "table" and attr_map.get("id") == "SHCtable":
            self.in_table = True
        elif self.in_table and tag == "tr" and "bodyrow" in classes:
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.in_detail = False
            self.main_parts = []
            self.detail_parts = []
        elif self.in_cell and tag == "div" and "detail" in classes:
            self.in_detail = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self.in_detail:
            self.in_detail = False
        elif tag == "td" and self.in_cell:
            main_text = re.sub(r"\s+", " ", "".join(self.main_parts)).strip()
            detail_text = re.sub(r"\s+", " ", "".join(self.detail_parts)).strip()
            self.current_row.append((main_text, detail_text))
            self.in_cell = False
            self.in_detail = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if not self.in_cell:
            return
        if self.in_detail:
            self.detail_parts.append(data)
        else:
            self.main_parts.append(data)


def parse_Massey_ratings(html_path: str, year: int = 1991) -> pd.DataFrame:
    """
    Parses a saved Massey Ratings HTML page (such as the 1991 FBS file)
    into a pandas DataFrame matching the 19-column massey_ratings schema.
    """
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    parser = _MasseyTableParser()
    parser.feed(html_content)

    parsed_data: list[list[str]] = []
    for cells in parser.rows:
        # Cell 0: Team is the main cell text; Conference is in <div class="detail">
        team_name, conference = cells[0]
        row_data = [team_name, conference]

        # Cells 1..N: primary value and nested .detail value for each column
        for main_val, detail_val in cells[1:]:
            row_data.append(main_val)
            row_data.append(detail_val)

        parsed_data.append(row_data)

    columns = [
        "Team", "Conference", "Record", "Win Pct",
        "Rank 1", "Detail 1", "Rank 2", "Detail 2",
        "Rank 3", "Detail 3", "Rank 4", "Detail 4",
        "Rank 5", "Detail 5", "Rank 6", "Detail 6",
        "Rank 7", "Detail 7",
    ]

    df = pd.DataFrame(parsed_data)
    df = df.reindex(columns=range(len(columns)), fill_value="")
    df.columns = columns
    df["Year"] = year
    return df