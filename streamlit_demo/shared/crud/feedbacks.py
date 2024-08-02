from loguru import logger

from shared.models import Feedback
from shared.models.engine import Session


class FeedbackCRUD:
    def __init__(self, session: Session):
        self.session = session

    def create(self, image_path: str, data: dict) -> bool:
        existed_feedback = self.get_by_image_path(image_path)
        if existed_feedback:
            self.delete_by_id(existed_feedback.id)
            logger.info(f"Image path: {image_path} exists. Deleted")

        feedback = Feedback(image_path=image_path, data=data)
        self.session.add(feedback)
        self.session.commit()

        logger.info(f"Added 1 row")
        return True

    def delete_by_id(self, feedback_id: int) -> bool:
        (
            self.session
            .query(Feedback)
            .filter(Feedback.id == feedback_id)
            .delete(synchronize_session=False)
        )
        return True

    def get_by_image_path(self, image_path: str) -> Feedback | None:
        result = (
            self.session
            .query(Feedback)
            .filter(Feedback.image_path == image_path)
            .first()
        )

        return result
