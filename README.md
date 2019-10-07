# Простой пример использования языка Julia для анализа данных


## Установка необходимых библиотек

### Способ 1

Выполнить файл `install_deps.jl`.

### Способ 2

Установить зависимости вручную в следующем порядке:

```
julia>]
pkg> add DataFrames
pkg> add CSV
pkg> add Conda
pkg> add ScikitLearn
pkg> add XGBoost
pkg> ^C (нужно нажать Ctrl-C)
julia> ENV["PYTHON"]=""
""
julia> using Pkg
julia> Pkg.build("PyCall")
julia> using Conda
julia> Conda.add("scikit-learn")
```

## Запуск

Выполнить файл `Titanic.jl`.