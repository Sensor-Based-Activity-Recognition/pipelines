# Stages
Stages werden in der dvc.yaml Datei erstellt. Die Stages verwenden die Python Skripte aus dem Ordner 'src', welche Modulare Funktionalitäten anbieten.
## DAG 
Um eine Übersicht über die Pipeline zu erhalten, kann der Befehl 'dvc dag' verwendet werden.
TODO: Add DAG Image
## Skripts für Tabulare Daten
Hier werden alle Skripts aufgelistet, welche Tabulare Daten (.parquet) als Input verwenden.
| Skript             | Beschreibung                                         | Output Typ       |
| ------------------ | ---------------------------------------------------- | ---------------- |
| min-max-scaler.py  | Scaliert die Daten auf einen bestimmten Wertebereich | Tabular + Params |
| pca.py             | Führt ein PCA auf den Daten durch                    | Tabular          |
| pull_data.py       | Holt die gesammelten Daten von der Datenbank         | Tabular          |
| resample.py        | Resamplet die Daten auf eine bestimmte Frequenz      | Tabular          |
| segmentate.py      | Segmentiert die Daten in bestimmte Zeitfenster       | Segments         |
| standard-scaler.py | Scaliert die Daten auf ihren z-Wert                  | Tabular + Params |

## Skripts für Segmentierte Daten
Hier werden alle Skripts aufgelistet, welche Segmentierte Daten (.dill) als Input verwenden.
| Skript                | Beschreibung                                        | Output Typ   |
| --------------------- | --------------------------------------------------- | ------------ |
| fft.py                | Führt eine FFT auf den Daten durch                  | Segmente     |
| aggregate.py          | Aggregiert die Daten anhand einer Aggregationsliste | Segmente     |
| correlations.py       | Berechnet die Korrelationen zwischen den Features   | Tabular      |
| train_test_split.py   | Teilt die Daten in Trainings- und Testdaten auf     | JSON Element |
| butterworth_filter.py | Filtert die Daten mit einem Butterworth Filter      | Segmente     |

## Skripts für Tabulare und Segmentierte Daten
Hier werden alle Skripts aufgelistet, welche Tabulare oder Segmentierte Daten als Input verwenden können.
TODO: @toeben1 Add when done

## Skripts für Trainieren und Evaluieren von Modellen
Hier werden alle Skripts aufgelistet, welche für das Trainieren und Evaluieren von Modellen verwendet werden.
| Skript          | Beschreibung                                       | Output Typ               |
| --------------- | -------------------------------------------------- | ------------------------ |
| model_runner.py | Führt ein Modell aus                               | Metriken und Vorhersagen |
| evaluate.py     | Erstellt anhand der Metriken alle benötigten Plots | Plots                    |