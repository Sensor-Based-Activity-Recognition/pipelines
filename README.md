# pipelines
Pipelines ist unser Repository für unsere Data-Processing-Pipelines und unseren trainierten Modelle. 
Die Pipelines werden mit DVC verwaltet und können mit DVC reproduziert werden.

## Kollaboration
Um neue Funktionen zu implementieren, muss ein neuer Branch erstellt werden.
Dieser soll danach von einem anderen Teammitglied reviewed werden.
Nicht funktionierende Pipelines (Fehler im Workflow) sollen nicht gemerged werden.
Daten müssen nicht unbedingt mit DVC gepusht werden, da der Workflow dies bereits übernimmt.

## DVC
DVC ist ein Tool von Iterative.ai, welches es ermöglicht, Pipelines zu verwalten und zu reproduzieren. 
Dieses wird mit Iterative Studio verknüpft, um unsere Modelle via Web-Interface zu verwalten.
Zusätzlich wird nach jedem Commit anhand von CML ein Workflow gestartet, welcher die Pipeline reproduziert und die Ergebnisse in Iterative Studio hochlädt.

### Stages
Die Pipelines bestehen aus Stages. Eine Stage ist eine einzelne Aufgabe, welche in einer Pipeline ausgeführt wird.
Eine Auflistung zu den Stages kann man unter [src/README.md](src/README.md) finden.

#### dvc.yaml and dvc.lock
Die dvc.yaml ist die Konfigurationsdatei für DVC. In dieser Datei wird die Pipeline anhand der Stages definiert.
Um die Reihenfolge der Pipeline zu ändern oder neue Stages hinzuzufügen, muss diese Datei angepasst werden.

Die dvc.lock Datei ist die Datei, welche die Abhängigkeiten der Stages definiert. Diese Datei wird automatisch von DVC generiert und sollte nicht manuell bearbeitet werden. Falls es zu Problemen kommt, kann diese Datei gelöscht werden und mit `dvc repro` neu generiert werden.

### Using DVC
Um sich mit DVC vertraut zu machen, hat Iterative.ai eine gute Zusammenfassung geschrieben:
https://dvc.org/doc/start

#### Most used commands
Hier ist eine kurze Zusammenfassung der wichtigsten DVC Befehle.

Installation von DVC mittels pip:
```bash
pip install dvc
```

Initialisierung eines DVC Repositories:
```bash
dvc init
```

Pullen der Daten (dvc):
```bash
dvc pull
```

Reproduzieren der Pipeline:
```bash
dvc repro
```

Pushen der Daten (dvc):
```bash
dvc push
```

Hinzufügen einer Stage:
```bash
dvc stage add -n <stage-name> -p <used-parameters-from-params-yaml> -d <dependencies> -o <outputs> python <script.py> <argvs>
```
Eine Stage darf auch manuell im dvc.yaml File erfasst werden. Dafür kann man sich an den anderen Stages orientieren.

### Parameters
Die Parameter werden in der params.yaml Datei definiert. Diese können danach im Iterative Studio oder manuell angepasst werden.

## Workflow
Nach jedem Commit wird ein Workflow gestartet, welcher die Pipeline reproduziert und die Ergebnisse in Iterative Studio hochlädt.
Die Beschreibung der Workflows kann unter [.github/workflows/README.md](.github/workflows/README.md) gefunden werden.

## Requirements
Nach dem hinzufügen einer Stage, soll überprüft werden, ob neue Python Packages hinzugefügt wurden.
Falls ja, müssen diese in der requirements.txt Datei mit der dazugehörigen Version hinzugefügt werden.

## Models and Results
Alle Resultate unserer Modelle werden unter [Modelle.md](https://github.com/Sensor-Based-Activity-Recognition/docs/blob/doc_patch/Modelle.md) dokumentiert.
