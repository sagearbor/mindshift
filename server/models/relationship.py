"""Pydantic models for the relationship graph model."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    couple = "couple"
    parent_child = "parent_child"
    coach_team = "coach_team"
    org = "org"
    custom = "custom"


class Participant(BaseModel):
    id: str
    role: str
    display_name: str
    parent_id: Optional[str] = None


class Relationship(BaseModel):
    id: str
    type: RelationshipType
    name: str
    participants: list[Participant] = []


class RelationshipEdge(BaseModel):
    from_participant_id: str
    to_participant_id: str
    context: str


class SessionContext(BaseModel):
    relationship_id: str
    edge: RelationshipEdge
    empathy_slider: int = Field(ge=0, le=100, default=50)


# --- Request/Response models for API ---

class RelationshipCreate(BaseModel):
    type: RelationshipType
    name: str
    participants: list[Participant]


class RelationshipOut(BaseModel):
    id: str
    type: RelationshipType
    name: str
    participants: list[Participant]
    created_at: str


class EdgeOut(BaseModel):
    from_participant_id: str
    from_display_name: str
    to_participant_id: str
    to_display_name: str
    context: str


class RelationshipSessionCreate(BaseModel):
    from_participant_id: str
    to_participant_id: str
    empathy_slider: int = Field(ge=0, le=100, default=50)
    metadata: dict = {}


class RelationshipSessionOut(BaseModel):
    id: str
    relationship_id: str
    from_participant_id: str
    to_participant_id: str
    edge_context: str
    empathy_slider: int
    created_at: str
    turns: list[dict]
    metadata: dict
