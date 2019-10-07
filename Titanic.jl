# Вспомогательные функции для вывода

"Печать заголовка раздела"
function startSection(title)
    border = repeat("=", length(title))
    println()
    println(border)
    println(title)
    println(border)
end

"Печать первых строк датасета"
function printHeads(train, test)
    println()
    println("Тренировочные данные")
    println(train[1:3, :])
    println()
    println("Тестовые данные")
    println(test[1:3, :])
end

startSection("Загрузка данных")

# Документация по DataFrames: https://juliadata.github.io/DataFrames.jl/stable/
using DataFrames
using CSV
using Statistics

# Загрузка наборов данных из файлов
train = CSV.read("input/train.csv", copycols=true)
test = CSV.read("input/test.csv", copycols=true)

# Вывод размеров датасетов
println("train shape: ", size(train), ", test shape: ", size(test))

printHeads(train, test)

#--------------------------------------------------------------------------------

startSection("Статистика по полам")

# Обращение к столбцам
#col = :Embarked
col = :Sex

# Объединение двух сгруппированных датасетов
# (by --- функция группировки, nrow считает количество строк в группе)
stat = hcat(by(train, col, nrow), by(test, col, nrow)[!, 2], makeunique = true)
# Переименование столбцов
names!(stat, [col, :train, :test])
# Добавление столбцов :p1 и :p2 с данными в процентах
insertcols!(stat, 3, :p1=>stat[!, :train]/size(train)[1]*100)
insertcols!(stat, 5, :p2=>stat[!, :test]/size(test)[1]*100)

println(stat)

startSection("Статистика по выжившим в классах обслуживания")

col = :Pclass
target = :Survived
println(sort(aggregate(train[:, [col, target]], col, mean), col))

#--------------------------------------------------------------------------------

startSection("Обработка отсутствующих значений")

println("Статистика по количеству отсутствующих значений в датасетах")
println(vcat(mapcols(x -> sum(ismissing.(x)), train),
             mapcols(x -> sum(ismissing.(x)), test), cols=:union))

"Заполнение отсутствующих значений медианными"
function fill_missing_by_median!(df, col)
    recode!(df[!, col], missing => median(skipmissing(df[!, col])))
end

for df in [train, test]
    fill_missing_by_median!(df, :Age)
    fill_missing_by_median!(df, :Fare)
    recode!(df[!, :Embarked], missing => "S")
    recode!(df[!,:Cabin], missing => "X")
end

println()
println("Статистика по количеству отсутствующих значений в датасетах (после обработки)")
println(vcat(mapcols(x -> sum(ismissing.(x)), train),
             mapcols(x -> sum(ismissing.(x)), test), cols=:union))

# Изменение типов столбцов (ранее там допускались Missing, но теперь их нет)
for df in [train, test]
    df[!, :Age] = convert.(Float64, df[!, :Age])
    df[!, :Fare] = convert.(Float64, df[!, :Fare])
end

printHeads(train, test)

#--------------------------------------------------------------------------------

startSection("Преобразование и удаление нечисловых данных")

function computeFare(str)
    c = str[1]
    if occursin(c, "ADET") # M
        return 2
    elseif occursin(c, "BC") # H
        return 3
    elseif occursin(c, "FG") # L
        return 1
    else
        return 0
    end
end

computeSex(str) =
    if str == "male"
        return 1
    else
        return 2
    end

for df in [train, test]
    # Преобразуем :Embarked в целое число
    recode!(df[!, :Embarked], df[!, :Embarked], "S" => "0", "C" => "1", "Q" => "2")

    # Точка после имени функции называется "broadcasting".
    # В результате функция применяется к каждому элементу массива
    df[!, :Embarked] = parse.(Int, df[!, :Embarked])

    # Преобразуем :Sex в целое число
    df[!, :Sex] = computeSex.(df[!, :Sex])

    # Преобразуем :Cabin в целое число
    df[!, :Cabin] = computeFare.(df[!, :Cabin])

    # Удаляем столбцы :Name и :Ticket
    select!(df, Not([:Name, :Ticket]))
end

printHeads(train, test)

#--------------------------------------------------------------------------------

startSection("Переход от датафреймов к массивам")

# Создаём обычные массивы
y = Array(train[!, :Survived])
X = Matrix(select(train, Not([:Survived])))

# Запоминаем названия столбцов в X
X_names = string.(names(select(train, Not([:Survived]))))

#--------------------------------------------------------------------------------

startSection("Логистическая регрессия")

# Документация: https://scikitlearnjl.readthedocs.io/en/latest/
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

# Разбиваем тренировочные данные на "новые" тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)

@sk_import preprocessing: StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@sk_import linear_model: LogisticRegression

# Создание и обучение модели
lr = LogisticRegression(random_state=21, solver="sag", max_iter=1000).fit(X_train_scaled, y_train)

# Предсказываем выживаемость
y_pred = lr.predict(X_test_scaled)

@sk_import metrics: accuracy_score

"Вывод значения точности предсказания"
function printAccuracy(prefix, y_test, y_pred)
    println("accuracy_", prefix, " = ", accuracy_score(y_test, y_pred))
end

printAccuracy("lr", y_test, y_pred)

#--------------------------------------------------------------------------------

startSection("Применение градиентного бустинга")

# Документация: https://xgboost.readthedocs.io/en/latest/index.html
using XGBoost

# Создание и обучение модели
num_round = 20
bst = xgboost(X_train, num_round, label = y_train,
              eta = 0.5, max_depth = 2, objective = "binary:logistic")

# Предсказываем вероятности выживаемости
preds_proba = XGBoost.predict(bst, X_test)

# Вычисляем 0 или 1 (выживаемость) округлением значений вероятностей
preds_class = round.(preds_proba)

printAccuracy("xgb", y_test, preds_class)

#--------------------------------------------------------------------------------

startSection("Определение оказывающих наибольшее влияние параметров")

print(importance(bst, X_names))

#--------------------------------------------------------------------------------

startSection("Запуск моделей на исходных тестовых данных и подготовка результатов для Kaggle")

function prepareSubmission(fname, test, y_pred)
    subm = DataFrame(PassengerID = test[:, :PassengerId], Survived = Int.(y_pred))
    CSV.write(fname, subm)
    println("Файл ", fname, " создан")
end

# Преобразуем тестовые данные в матрицу и избавляемся от Missing в типе
full_test = convert.(Float64, Matrix(test))

# Пользуемся логистической регрессией
y_pred = lr.predict(full_test)
prepareSubmission("titanic_submission_lr.csv", test, y_pred)

# Пользуемся градиентным бустингом
y_pred = round.(XGBoost.predict(bst, full_test))
prepareSubmission("titanic_submission_xgb.csv", test, y_pred)