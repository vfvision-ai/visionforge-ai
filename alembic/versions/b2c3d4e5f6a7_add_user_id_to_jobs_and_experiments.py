"""add_user_id_to_jobs_and_experiments

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-19 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('training_jobs', sa.Column('user_id', sa.String(), nullable=True))
    op.create_index('ix_training_jobs_user_id', 'training_jobs', ['user_id'])

    op.add_column('experiments', sa.Column('user_id', sa.String(), nullable=True))
    op.create_index('ix_experiments_user_id', 'experiments', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_experiments_user_id', table_name='experiments')
    op.drop_column('experiments', 'user_id')

    op.drop_index('ix_training_jobs_user_id', table_name='training_jobs')
    op.drop_column('training_jobs', 'user_id')
