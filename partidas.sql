CREATE TABLE partidas_brasileirao (
    data DATE,
    horario TIME,
    id_time_mandantes INTEGER REFERENCES times_brasileiros(id),
    time_mandante TEXT,
    resultado TEXT,
    id_time_visitante INTEGER REFERENCES times_brasileiros(id),
    time_visitante TEXT,
    publico INTEGER,
    estadio TEXT
);
