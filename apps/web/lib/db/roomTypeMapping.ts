import { RoomType as DbRoomTypeEnum } from './schema';
import { RoomType as ClientRoomType } from '@3dix/types';

type DbRoomType = (typeof DbRoomTypeEnum)[keyof typeof DbRoomTypeEnum];

export const dbToClientRoomType: Record<DbRoomType, ClientRoomType> = {
  bedroom: 'bedroom',
  kitchen: 'kitchen',
  bathroom: 'bathroom',
  closet: 'closet',
  living_room: 'living',
  dining_room: 'dining',
  custom: 'custom',
};

export const clientToDbRoomType: Record<ClientRoomType, DbRoomType> = {
  bedroom: DbRoomTypeEnum.BEDROOM,
  kitchen: DbRoomTypeEnum.KITCHEN,
  bathroom: DbRoomTypeEnum.BATHROOM,
  closet: DbRoomTypeEnum.CLOSET,
  living: DbRoomTypeEnum.LIVING_ROOM,
  dining: DbRoomTypeEnum.DINING_ROOM,
  custom: DbRoomTypeEnum.CUSTOM,
};

type WithClientRoomType<T> = Omit<T, 'roomType'> & { roomType: ClientRoomType };

export function mapDbRoomType(roomType: string | null | undefined): ClientRoomType {
  if (!roomType) {
    return 'custom';
  }

  return dbToClientRoomType[roomType as DbRoomType] ?? 'custom';
}

export function mapClientRoomType(roomType: ClientRoomType): DbRoomType {
  return clientToDbRoomType[roomType];
}

export function mapRoomFromDb<T extends { roomType: string | null | undefined }>(room: T): WithClientRoomType<T> {
  return {
    ...room,
    roomType: mapDbRoomType(room.roomType),
  };
}

export function mapRoomsFromDb<T extends { roomType: string | null | undefined }>(rooms: T[]): WithClientRoomType<T>[] {
  return rooms.map((room) => mapRoomFromDb(room));
}
