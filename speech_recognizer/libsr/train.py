from keras.optimizers import SGD


def train(model, x, y, epochs, batch_size, *,
          optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
          loss='categorical_crossentropy',
          metrics=['accuracy']):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(x, y, epochs=5, batch_size=32)
    return model


def train_generator(model, generator, steps_per_epoch, epochs=1, *,
                    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'], **kwargs):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit_generator(generator, steps_per_epoch, epochs=epochs, **kwargs)
    return model
