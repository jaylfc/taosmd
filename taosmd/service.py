async def a2a_create_thread(
    thread: str,
    participants: list[str],
    agent: str,
    data_dir=None,
) -> dict:
    """Create an A2A thread with initial participants.

    The caller (``agent``) is added as an owner by default. ``participants``
    (excluding the caller) are added as members. If no participants are
    provided, creates a DM with exactly two members: the caller and one
    other principal (not supported by this implementation - requires participants).

    Returns the thread and the list of its active members after creation.
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_create_thread(
            thread, participants, agent, data_dir,
        )

    from .a2a_membership import MembershipStore

    store = MembershipStore(data_dir)
    try:
        # Validate thread name (alphanumeric and hyphens only)
        if not participants:
            raise ValueError("participants list cannot be empty")

        # Check if thread already exists (by checking for any membership)
        if store.has_any_membership(thread):
            raise ValueError(f"thread '{thread}' already exists")

        # Create membership records
        # Creator is owner
        await store.add_membership(thread, agent, role="owner")
        # Participants are members (excluding the creator if already in participants)
        for participant in participants:
            if participant != agent:  # Avoid duplicate
                await store.add_membership(thread, participant, role="member")

        # Archive the creation
        ts = time.time()
        await store.archive_membership_created(thread, agent, "owner", ts, data_dir)
        for participant in participants:
            if participant != agent:
                await store.archive_membership_created(thread, participant, "member", ts, data_dir)

        # Get active members
        members = await store.list_active_members(thread)
        return {
            "thread": thread,
            "created": True,
            "active_members": [
                {"principal_id": m.principal_id, "role": m.role, "created_at": m.created_at}
                for m in members
            ],
        }
    finally:
        await store.close()


async def a2a_list_members(
    thread: str,
    data_dir=None,
) -> list[dict]:
    """List all active members (owners and members) of a thread.

    Excludes threads that have no membership records (open/legacy threads).
    """
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_list_members(thread, data_dir)

    from .a2a_membership import MembershipStore

    store = MembershipStore(data_dir)
    try:
        if not store.has_any_membership(thread):
            # Legacy thread with no membership rows is open to all
            # Return empty list to indicate open thread (backward compatibility)
            return []

        members = await store.list_active_members(thread)
        return [
            {
                "principal_id": m.principal_id,
                "role": m.role,
                "created_at": m.created_at,
            }
            for m in members
        ]
    finally:
        await store.close()


async def a2a_add_member(
    thread: str,
    principal_id: str,
    agent: str,
    data_dir=None,
) -> dict:
    """Add a member to a thread. Caller must be an owner."""
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_add_member(thread, principal_id, agent, data_dir)

    from .a2a_membership import MembershipStore

    store = MembershipStore(data_dir)
    try:
        # Check if principal is already an active member
        existing = await store.get_membership(thread, principal_id)
        if existing is not None:
            return {
                "thread": thread,
                "principal_id": principal_id,
                "added": False,
                "already_member": True,
            }

        # Verify caller is an owner
        if not await store.is_principal_owner(thread, agent):
            raise PermissionError(f"caller '{agent}' is not an owner of thread '{thread}'")

        # Add as member
        await store.add_membership(thread, principal_id, role="member")

        # Archive the addition
        ts = time.time()
        await store.archive_membership_created(thread, principal_id, "member", ts, data_dir)

        return {
            "thread": thread,
            "principal_id": principal_id,
            "added": True,
        }
    finally:
        await store.close()


async def a2a_remove_member(
    thread: str,
    principal_id: str,
    agent: str,
    data_dir=None,
) -> dict:
    """Remove a member from a thread. Caller must be an owner and cannot remove the last owner."""
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.a2a_remove_member(thread, principal_id, agent, data_dir)

    from .a2a_membership import MembershipStore

    store = MembershipStore(data_dir)
    try:
        # Verify caller is an owner
        if not await store.is_principal_owner(thread, agent):
            raise PermissionError(f"caller '{agent}' is not an owner of thread '{thread}'")

        # Cannot remove if this would leave the thread without any owners
        if await store.is_principal_owner(thread, principal_id):
            owners = await store.get_thread_owners(thread)
            if len(owners) <= 1:
                raise ValueError(
                    f"cannot remove the last owner of thread '{thread}'; "
                    "transfer ownership or add another owner first"
                )

        # Remove membership (mark inactive)
        removed = await store.remove_membership(thread, principal_id)
        if not removed:
            return {
                "thread": thread,
                "principal_id": principal_id,
                "removed": False,
                "not_found": True,
            }

        # Archive the removal
        ts = time.time()
        await store.archive_membership_removed(thread, principal_id, ts, data_dir)

        return {
            "thread": thread,
            "principal_id": principal_id,
            "removed": True,
            "archived": True,
        }
    finally:
        await store.close()
