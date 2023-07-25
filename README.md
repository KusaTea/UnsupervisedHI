# UnsupervisedHI
Unsupervised approach to Health Index estimation​

Описание файлов:<ul>
<li>RUL_adding_ipynb - код для выделения колонки RUL</li>
<li>DEA_Visualization - очистка датасета</li>
<li>ae_metrics - класс метрики Mean Absolute Percentage Error</li>
<li>ae_training - базовое обучение MLP автокодировщика</li>
<li>ae_training_normal - обучение автокодировщика на нормальных данных и сравнние ошибки восстановления нормальных и аномальных данных</li>
<li>construct_dataset - создание датасета (совмещает в себе функционал RUL_adding_ipynb и DEA_Visualization)</li>
<li>custom_classes - классы автокодировщика и датасета, и функции разделения датасетов</li>
<li>normalisation - методы нормализации данных</li>
<li>rul_metrics - метрики для сравнения истиных и предсказанных кривых HI и значений RUL</li>
</ul>
