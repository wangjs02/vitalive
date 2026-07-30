# VitalDB Data Notes

## Server Data Root

VitalDB data should live outside the `prod` repository so it can be shared by
multiple projects.

```text
/home/junshi/data/VitalDB/
├── docs/
├── metadata/
├── zip/
├── raw/
└── processed/
```

Papers and dataset documentation belong in `~/data/VitalDB/docs/`. Metadata
tables belong in `~/data/VitalDB/metadata/`. Raw `.vital` files belong in
`~/data/VitalDB/raw/`. Downloaded OSF component archives belong in
`~/data/VitalDB/zip/`. Processed outputs should be written under
`~/data/VitalDB/processed/<method>/`.

## OSF Vital Files

The OSF project for VitalDB vital files is:

```text
https://osf.io/dtc45/overview
```

It is split into three components:

```text
Vital Files 0001-2000: component 49etp, 2000 files, 30.0 GB
Vital Files 2001-4000: component wxq57, 2000 files, 30.4 GB
Vital Files 4001-6388: component nsge2, 2388 files, 36.9 GB
```

Known component zip URLs:

```text
https://files.de-1.osf.io/v1/resources/49etp/providers/osfstorage/?zip=
https://files.de-1.osf.io/v1/resources/wxq57/providers/osfstorage/?zip=
https://files.de-1.osf.io/v1/resources/nsge2/providers/osfstorage/?zip=
```

Use the OSF API to discover each component ID and its zip download URL:

```bash
python3 -c '
import requests

project_id = "dtc45"
url = f"https://api.osf.io/v2/nodes/{project_id}/children/"
res = requests.get(url).json()

for child in res.get("data", []):
    title = child["attributes"]["title"]
    node_id = child["id"]
    print(f"发现组件: {title} (ID: {node_id})")
    zip_url = f"https://files.de-1.osf.io/v1/resources/{node_id}/providers/osfstorage/?zip="
    print(f"下载链接: {zip_url}")
'
```

On the server, download component archives into `~/data/VitalDB/zip/`, then
extract them into `~/data/VitalDB/raw/` before running preprocessing.
