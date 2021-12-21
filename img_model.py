from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()



class EntityBase(object):
    def to_json(self):
        fields = self.__dict__
        if "_sa_instance_state" in fields:
            del fields["_sa_instance_state"]
        return fields


class Img(db.Model, EntityBase):
    __tablename__ = 'ic_tbl'
    ic_id = db.Column(db.Integer, primary_key=True)
    ic_caption = db.Column(db.String(200))


if __name__ == "__main__":
    imgs = Img.query.all()
    print(imgs)
    imgs_output = []
    for img in imgs:
        imgs_output.append(img.to_json())
    print(imgs_output);