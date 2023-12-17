# KI Abgabe &mdash; Connect 4 using Evolutionary

Repository: https://github.com/NekkoDroid/AbgabeKI

Das gesamte Projekt kann an innerhalb der [`__main__.py`] Datei finden. Diese besteht aus einem sehr simplen "Connect 4" (deutsch. "4 Gewinnt"),
einem Neural Network der mit Tensorflow gemacht wurde, als auch ein Algorithmus, der zur Genetischen Evolution des Neural Networks mehrerer
Generationen benutzt wird. Ziel diese Projektes ist es eine Model zu kreieren, gegen welches man Spielen kann und das in so wenig Zügen wie möglich
gewinnt und möglichs langsam verliert wenn dies zutrifft.

Im dieser "Connect 4" implementation gib es kein unentschieden. Wenn es dazu kommt, dass das Spielfeld ausgefüllt ist, verliert der Spieler, der
keinen Chip mehr spielen kann.

## Der Code

Das Model wird in der `create_neural_network` beschrieben und lässt sich dort auch anpassen.

`check_fitness` beschreibt die fitness funktion die versucht wird zu maximieren. Um immer auf gewinne auszuziehlen geben diese immer mehr als
verluste, dabei werden schnellere gewinne mehr belohnt (mit theoretischem maximum von `ROW_COUNT * COL_COUNT` wenn der erste zug vom gegner auserhalb platziert wird,
praktisch ist dies jedoch `ROW_COUNT * COL_COUNT / 7` da dies die minimum Anzahl an Zügen ist). Beim verlieren wird darauf optimiert, so lange herauszuspielen
um zu versuchen durch vollspielen zu gewinnen.

`mutate` dient einfach dazu  `MUTATION_RATE` an den verschiedenen weights anzuwenden.

Mit `crossover` kann man mit einer beliebigen Anzahl > 1 an Modelle kombinieren aus denen die Weights zufällig dem Kind zugewiesen werden.

`train_neural_network` trainiert die modelle an deren eigenen Spielen um deren zufälligen variationen zu minimieren.

In `main` besteht Großteil der Logik zur Generation Verwaltung. Es werden eine population nach deren fitness gegen andere individuen aus der generation
gerankt, woraus die bessere Hälfte in die nächste Generation übertragen werden und die andere Hälfte wird durch Kreuzung zufälliger übertragenen ersetzt.

### Konstanten

- `GAMES_FOR_AVERAGE` beschreibt wie viele Spiele mit einer generation getspielt werden um die durchschnittliche fitness zu bestimmen
- `GENERATIONS` beschreibt wie viele generationen gelaufen werden bis beendet wird
- `POPULATION` beschreibt wie viele modelle pro generation erstellt werden
- `MUTATION_RATE` beschreibt wie wahrscheinlich sich die weights mutieren

## Geplant

Eigentlich soll das Model auch aus `Conv2d` layer bestehen, jedoch ist beim einbinden vom `tf.keras.layers.Conv2D` in das model
Fehler aufgetaucht, die ich nicht beheben konnte, weshalb es komplett vorerst wegelassen wurde.

Aus irgend einem grund stagniert the durchschnittliche fitness bei `6.0`. Fitness von `6.0` heißt, dass ein Spiel in 7 runden gewonnen wurde.
Somit gehe ich davon aus, dass etwas in der `evaluate_best_individuals` falsch implementiert wurde/ein massiver denkfehler passiert ist.
(Ab und zu werdern andere Werte ausgegeben, weshalb ich unsicher bin)
