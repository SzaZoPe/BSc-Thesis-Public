import pandas as pd
import numpy as np
import joblib
import pyarrow
from sklearn.svm import SVC
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def get_luminance_qtable(jpeg_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(jpeg_path, 'rb') as f:
        img = Image.open(f)
        if hasattr(img, 'quantization'):
            qtables = img.quantization
            luminance_qtable = qtables.get(0)
            if luminance_qtable:
                return list(np.array(luminance_qtable).flatten())
    return None

if __name__ == "__main__":
    factors = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    data = []

    for q in factors:
        for i in range(1, 101):
            path = f"test_images/images/normal_jpegs/{i}_q{q}.jpg"
            luminance_qtable = get_luminance_qtable(path)
            data.append([q] + luminance_qtable)

    columns = ['q'] + [f'v{i}' for i in range(64)]
    df = pd.DataFrame(data, columns=columns)

    y_train = df['q']
    x_train = df.drop(columns='q')

    param = {
        'C': np.arange(1, 13, 1),
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    SVModel = SVC()
    GridS = GridSearchCV(SVModel, param, cv=5, n_jobs=-1, verbose=1)
    GridS.fit(x_train, y_train)

    print(GridS.best_params_)

    print(confusion_matrix(y_train, GridS.predict(x_train)))
    print(accuracy_score(y_train, GridS.predict(x_train)))

    joblib.dump(GridS, 'matrix.joblib')
