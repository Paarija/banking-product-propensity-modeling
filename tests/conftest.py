import pandas as pd
import pytest

from bankrec.data import PRODUCT_COLUMNS


@pytest.fixture
def tiny_santander():
    rows = []
    for client in range(8):
        for month in range(1, 6):
            row = {"ncodpers": str(client), "fecha_dato": f"2015-{month:02d}-28"}
            row.update(dict.fromkeys(PRODUCT_COLUMNS, 0))
            if month >= 2 and client % 2 == 0:
                row[PRODUCT_COLUMNS[0]] = 1
            if month >= 4 and client % 2 == 1:
                row[PRODUCT_COLUMNS[1]] = 1
            if month >= 5 and client % 3 == 0:
                row[PRODUCT_COLUMNS[2]] = 1
            rows.append(row)
    return pd.DataFrame(rows)
